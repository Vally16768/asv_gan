# train.py — ASV-GAN strict + WGAN-R1 + evasion loss + annealing + samples/metrics per epoch + best-at-epoch
from __future__ import annotations
from pathlib import Path
import os, json, csv, time
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torchaudio
import joblib

from scipy.special import softmax

# ----------------- Proiect -----------------
from constants import (
    # audio/feats
    SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    FEATS_MEAN, FEATS_STD,
    # train
    AMP_ENABLED, BATCH_SIZE, EPOCHS, CRITIC_ITERS,
    LR_G, LR_D, BETA1, BETA2,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_R1,
    LOG_INTERVAL, VAL_INTERVAL, SAVE_DIR, ROOT,
    # Schedules
    DELTA_INIT, DELTA_MIN, DELTA_DECAY,
    INST_NOISE_INIT, INST_NOISE_MIN, INST_NOISE_DECAY,
    # Early stop
    EARLY_STOP_ENABLED, EARLY_STOP_PATIENCE, EARLY_STOP_METRIC,
    # ASVspoof
    ASV_MODEL_DIR, ASV_SCALER, ASV_SR, ASV_COMBO,
    EVASION_LAMBDA, EVASION_EVERY, TARGET_LABEL,
)
from dataset import ASVBonafideDataset, pad_collate
from models import Generator, Critic
from losses import wgan_g_loss, wgan_d_loss, r1_regularizer
from utils import set_seed

# detector keras (obligatoriu)
from detector_keras import load_keras_model

# tqdm (optional)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# ----------------- Seed & device -----------------
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Helpers stabilitate -----------------
def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

# ---------- Mel (device-aware) & Griffin-Lim ----------
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    center=True,
    pad_mode="reflect",
    power=1.0,
    norm="slaney",
    mel_scale="htk",
).to(device)

def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """
    [B,1,T] / [B,T] -> logmel [B,M,Tf]
    """
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    wave = wave.to(device)
    mel = _mel(wave)
    mel = torch.log1p(torch.clamp(mel, min=0.0))
    return _safe(mel)

class MelToWave(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SR,
        )
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            power=1.0,
            n_iter=60
        )

    @torch.no_grad()
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        logmel [B,M,Tf] -> wave [B,1,T]
        """
        mel = torch.expm1(logmel).clamp(min=0.0)
        mag = self.inv_mel.to(logmel.device)(mel)
        wave = self.griffin.to(logmel.device)(mag)
        return wave.unsqueeze(1)

mel2wav = MelToWave().to(device)

# ---------- Vector ASV 61-D (caching DCT) ----------
_DCT_CACHE = {}
def _delta_along_time(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("Expected [B,C,T] for delta computation")
    C = x.shape[1]
    k = torch.tensor([[-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3)
    y = F.pad(x, (1, 1), mode="replicate")
    y = F.conv1d(y, k.expand(C, 1, 3), groups=C) / 2.0
    return y

def _mfcc_from_logmel(logmel: torch.Tensor, n_mfcc: int = 20) -> torch.Tensor:
    import math
    B, M, T = logmel.shape
    key = (M, n_mfcc, logmel.device, logmel.dtype)
    if key not in _DCT_CACHE:
        m = torch.arange(M, device=logmel.device, dtype=logmel.dtype)
        k = torch.arange(M, device=logmel.device, dtype=logmel.dtype).unsqueeze(1)
        scale = torch.sqrt(torch.tensor(2.0 / M, device=logmel.device, dtype=logmel.dtype))
        dct = scale * torch.cos((math.pi / M) * (m + 0.5) * k)
        dct[0] = dct[0] / math.sqrt(2.0)
        _DCT_CACHE[key] = dct.t()   # [M,M]
    dctT = _DCT_CACHE[key]         # [M,M]
    x = logmel.transpose(1, 2)     # [B,T,M]
    mfcc = x @ dctT[:, :n_mfcc]    # [B,T,n_mfcc]
    return _safe(mfcc.transpose(1, 2))

@torch.no_grad()
def make_asv_vector_from_wave(wave: torch.Tensor) -> torch.Tensor:
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    logmel = logmel_from_wave(wave)                 # [B,M,Tf]
    mfcc  = _mfcc_from_logmel(logmel, n_mfcc=20)    # [B,20,Tf]
    d1    = _delta_along_time(mfcc)                 # [B,20,Tf]
    d2    = _delta_along_time(d1)                   # [B,20,Tf]
    energy = logmel.mean(dim=(1,2), keepdim=False)  # [B]
    vec = torch.cat([
        mfcc.mean(dim=-1), d1.mean(dim=-1), d2.mean(dim=-1),
        energy.view(-1, 1)
    ], dim=1).to(torch.float32)                     # [B,61]
    return vec

# ----------------- ASVspoof REQUIRE (no fallbacks) -----------------
def _require_file(p: Path, what: str):
    if not Path(p).exists():
        raise FileNotFoundError(f"[ASV REQUIRE] Missing {what}: {p}. Aborting.")

def _load_required_asv_assets():
    # model obligatoriu (keras .keras sau .h5)
    model_main = Path(ASV_MODEL_DIR) / "best_model.keras"
    model_alt  = Path(ASV_MODEL_DIR) / "best_model.h5"
    if model_main.exists():
        model_path = model_main
    elif model_alt.exists():
        model_path = model_alt
    else:
        raise FileNotFoundError(
            f"[ASV REQUIRE] Missing Keras model. Expected one of:\n"
            f"  - {model_main}\n  - {model_alt}"
        )
    keras_model = load_keras_model(model_path)

    # scaler obligatoriu
    _require_file(ASV_SCALER, "ASV scaler")
    scaler = joblib.load(str(ASV_SCALER))
    dim = None
    if hasattr(scaler, "mean_"):
        dim = int(np.array(scaler.mean_).size)
    elif hasattr(scaler, "scale_"):
        dim = int(np.array(scaler.scale_).size)
    if dim != 61:
        raise RuntimeError(f"[ASV REQUIRE] Scaler dim must be 61, got: {dim}")

    # labels & target
    labels_path = Path(ASV_MODEL_DIR) / "labels.txt"
    _require_file(labels_path, "ASV labels.txt")
    labels = [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]
    if TARGET_LABEL not in labels:
        raise RuntimeError(f"[ASV REQUIRE] TARGET_LABEL='{TARGET_LABEL}' not found in labels.txt ({labels_path}).")
    target_idx = labels.index(TARGET_LABEL)

    return keras_model, scaler, target_idx, labels

# ----------------- Training utils -----------------
def save_checkpoint(state: dict, name: str):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / name
    torch.save(state, path)
    return path

def spec_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b)

def save_audio_batch(prefix_dir: Path, waves: torch.Tensor, sr: int, max_items: int = 4, tag: str = "fake"):
    prefix_dir.mkdir(parents=True, exist_ok=True)
    B = waves.shape[0]
    n = min(B, max_items)
    w = waves.detach().cpu().clamp(-1, 1)
    for i in range(n):
        path = prefix_dir / f"{tag}_{i:02d}.wav"
        torchaudio.save(str(path), w[i], sr)

def write_hparams_json(path: Path, hparams: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(hparams, f, indent=2, sort_keys=True)

# ----------------- MAIN -----------------
def main():
    print("== ASV-GAN Training (strict) ==")
    print(f"Device: {device}")
    print(f"Dataset root: {ROOT}")

    # Require ASV assets
    keras_model, asv_scaler, target_idx, labels = _load_required_asv_assets()
    print(f"[ASV] Using model at: {ASV_MODEL_DIR}")
    print(f"[ASV] Using scaler:   {ASV_SCALER} (dim=61)  target='{TARGET_LABEL}' idx={target_idx}")

    # === Detectare head (sigmoid vs softmax) ===
    _probe = np.zeros((1, 61), dtype="float32")
    _probe = asv_scaler.transform(_probe)
    _pred = keras_model.predict(_probe, verbose=0)
    if _pred.ndim != 2:
        raise RuntimeError(f"[ASV REQUIRE] Modelul trebuie să dea un 2D array [B,C]. Am primit shape={_pred.shape}")

    num_classes = _pred.shape[1]
    if num_classes == 1:
        asv_head = "sigmoid"
        if len(labels) != 2:
            raise RuntimeError(f"[ASV REQUIRE] Pentru head sigmoid (C=1) labels.txt trebuie să aibă EXACT 2 clase; are {len(labels)}.")
        neg_label, pos_label = labels[0], labels[1]
        if TARGET_LABEL == pos_label:
            sigmoid_mapping = "target_is_pos"
        elif TARGET_LABEL == neg_label:
            sigmoid_mapping = "target_is_neg"
        else:
            raise RuntimeError(f"[ASV REQUIRE] TARGET_LABEL='{TARGET_LABEL}' nu se găsește în labels.txt.")
        print(f"[ASV] Head: sigmoid (C=1). Pos='{pos_label}', Neg='{neg_label}', map='{sigmoid_mapping}'.")
    else:
        asv_head = "softmax"
        if target_idx >= num_classes:
            raise RuntimeError(f"[ASV REQUIRE] target_idx={target_idx} depășește num_classes={num_classes}")
        print(f"[ASV] Head: softmax (C={num_classes}). TARGET idx={target_idx}")

    # Models
    G = Generator(c_in=N_MELS).to(device)
    D = Critic(c_in=N_MELS).to(device)

    # Opts
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    scaler = GradScaler(enabled=AMP_ENABLED)

    # LR schedulers (ReduceLROnPlateau pe val loss G)
    schedG = optim.lr_scheduler.ReduceLROnPlateau(optG, mode="min", factor=0.5, patience=2, verbose=True, min_lr=1e-6)
    schedD = optim.lr_scheduler.ReduceLROnPlateau(optD, mode="min", factor=0.5, patience=2, verbose=True, min_lr=1e-6)

    # Early stopping
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    # Schedules state
    delta_scale = float(DELTA_INIT)
    inst_sigma  = float(INST_NOISE_INIT)

    # Data
    train_ds = ASVBonafideDataset(split="train", use_validation=False)
    val_ds   = ASVBonafideDataset(split="val",   use_validation=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(device=="cuda"),
                              collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(device=="cuda"),
                            collate_fn=pad_collate)

    # Logs
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_DIR / "train_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "time","epoch","step","phase","lossD","lossG","g_gan","g_spec","r1",
                "lrG","lrD","delta_scale","inst_sigma","evasion","val_evasion"
            ])

    # hparams snapshot
    write_hparams_json(SAVE_DIR / "hparams.json", {
        "SR": SR, "N_MELS": N_MELS, "N_FFT": N_FFT, "HOP_LENGTH": HOP_LENGTH, "WIN_LENGTH": WIN_LENGTH,
        "AMP_ENABLED": AMP_ENABLED, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "CRITIC_ITERS": CRITIC_ITERS,
        "LR_G": LR_G, "LR_D": LR_D, "BETA1": BETA1, "BETA2": BETA2,
        "LAMBDA_GAN": LAMBDA_GAN, "LAMBDA_SPEC": LAMBDA_SPEC, "LAMBDA_R1": LAMBDA_R1,
        "EVASION_LAMBDA": EVASION_LAMBDA, "EVASION_EVERY": EVASION_EVERY, "TARGET_LABEL": TARGET_LABEL,
        "ASV_HEAD": asv_head,
        "DELTA_INIT": DELTA_INIT, "DELTA_MIN": DELTA_MIN, "DELTA_DECAY": DELTA_DECAY,
        "INST_NOISE_INIT": INST_NOISE_INIT, "INST_NOISE_MIN": INST_NOISE_MIN, "INST_NOISE_DECAY": INST_NOISE_DECAY,
        "VAL_INTERVAL(steps)": VAL_INTERVAL,
        "EARLY_STOP_METRIC": EARLY_STOP_METRIC
    })

    global_step = 0
    next_val_step = VAL_INTERVAL  # validare pe pași

    for epoch in range(EPOCHS):
        G.train(); D.train()

        # >>> Agregatori pe epocă (pentru linia "epoch" în CSV)
        ep_lossD_sum = 0.0
        ep_lossG_sum = 0.0
        ep_g_gan_sum = 0.0
        ep_g_spec_sum = 0.0
        ep_r1_sum = 0.0
        ep_evasion_sum = 0.0
        ep_evasion_cnt = 0
        ep_steps = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, batch in pbar:
            wave = batch["wave"].to(device)            # [B,1,T]
            mel_real = logmel_from_wave(wave)          # [B,M,Tf]

            # --------- Update Critic (x CRITIC_ITERS) ----------
            for _ in range(CRITIC_ITERS):
                with autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=AMP_ENABLED):
                    # delta controlată de scală + tanh-clipping
                    delta_raw = G(mel_real.detach())
                    delta = torch.tanh(delta_raw) * delta_scale
                    mel_fake = (mel_real.detach() + delta).clamp_min(-8.0)

                    # instance noise simetric pe intrarea lui D
                    if inst_sigma > 0.0:
                        noise_r = torch.randn_like(mel_real) * inst_sigma
                        noise_f = torch.randn_like(mel_fake) * inst_sigma
                        d_real = D(mel_real + noise_r)
                        d_fake = D(mel_fake.detach() + noise_f)
                    else:
                        d_real = D(mel_real)
                        d_fake = D(mel_fake.detach())

                    lossD = wgan_d_loss(d_real, d_fake)

                optD.zero_grad(set_to_none=True)
                scaler.scale(lossD).backward()

                # R1 pe real — fără autocast, cu requires_grad=True
                with autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=False):
                    mel_real_r1 = mel_real.detach().requires_grad_(True)
                    d_real_r1 = D(mel_real_r1)
                    r1 = r1_regularizer(d_real_r1, mel_real_r1)
                scaler.scale(LAMBDA_R1 * r1).backward()

                scaler.step(optD)
                scaler.update()  # update și după D

            # --------- Update Generator ----------
            with autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=AMP_ENABLED):
                delta_raw = G(mel_real)                                  # [B,M,Tf]
                delta = torch.tanh(delta_raw) * delta_scale
                mel_fake = (mel_real + delta).clamp_min(-8.0)
                d_fake = D(mel_fake)
                g_gan = wgan_g_loss(d_fake) * LAMBDA_GAN
                g_spec = spec_l1(mel_fake, mel_real) * LAMBDA_SPEC
                lossG = g_gan + g_spec

            # === Evasion loss (ASVspoof mandatory) ===
            evasion_mean = np.nan
            if EVASION_LAMBDA > 0.0 and (global_step % EVASION_EVERY) == 0:
                with torch.no_grad():
                    wave_fake = mel2wav(mel_fake)                     # [B,1,T]
                    asv_vec = make_asv_vector_from_wave(wave_fake)    # [B,61]
                x_np = asv_vec.detach().cpu().numpy().astype("float32")
                x_np = asv_scaler.transform(x_np)
                preds = keras_model.predict(x_np, verbose=0)          # [B,C]
                if preds.ndim != 2:
                    raise RuntimeError(f"[ASV] Pred shape invalid: {preds.shape}")

                if asv_head == "softmax":
                    probs = softmax(preds, axis=1)
                    p_target_np = probs[:, target_idx]
                else:  # sigmoid
                    p_sig = preds[:, 0].astype("float32")  # P(labels[1])
                    p_target_np = p_sig if sigmoid_mapping == "target_is_pos" else (1.0 - p_sig)

                evasion_mean = float(p_target_np.mean())
                evasion_loss = torch.tensor(evasion_mean, dtype=torch.float32, device=device)
                lossG = lossG + (EVASION_LAMBDA * evasion_loss)

                # >>> agregare pe epocă (doar când e calculată)
                ep_evasion_sum += evasion_mean
                ep_evasion_cnt += 1

            optG.zero_grad(set_to_none=True)
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()

            # ---- annealing schedules ----
            delta_scale = max(DELTA_MIN, delta_scale * DELTA_DECAY)
            inst_sigma  = max(INST_NOISE_MIN, inst_sigma * INST_NOISE_DECAY)

            # >>> agregare pe epocă (medii)
            ep_lossD_sum += float(lossD.detach().cpu())
            ep_lossG_sum += float(lossG.detach().cpu())
            ep_g_gan_sum += float(g_gan.detach().cpu())
            ep_g_spec_sum += float(g_spec.detach().cpu())
            ep_r1_sum    += float(r1.detach().cpu())
            ep_steps     += 1

            # Logs (pe pași)
            if (global_step % LOG_INTERVAL) == 0:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        int(time.time()), epoch, global_step, "train",
                        float(lossD.detach().cpu()), float(lossG.detach().cpu()),
                        float(g_gan.detach().cpu()), float(g_spec.detach().cpu()),
                        float(r1.detach().cpu()),
                        optG.param_groups[0]["lr"], optD.param_groups[0]["lr"],
                        delta_scale, inst_sigma, evasion_mean, np.nan
                    ])
                pbar.set_description(
                    f"E{epoch} S{global_step} | D {lossD:.3f} | G {lossG:.3f} | gg {g_gan:.3f} | gs {g_spec:.3f} | r1 {r1:.3f} | δ {delta_scale:.4f} σ {inst_sigma:.3f} | ev {evasion_mean if not np.isnan(evasion_mean) else -1:.3f}"
                )

            # --------- Validare pe PAȘI (opțional) ----------
            if global_step > 0 and global_step >= next_val_step:
                G.eval(); D.eval()
                with torch.no_grad():
                    val_ok = False
                    try:
                        batch_val = next(iter(val_loader))
                        val_ok = True
                    except StopIteration:
                        # dataset val gol — ignorăm validarea pe pași
                        val_ok = False

                    if val_ok:
                        wave_val = batch_val["wave"].to(device)
                        mel_real_v = logmel_from_wave(wave_val)
                        delta_v = torch.tanh(G(mel_real_v)) * delta_scale
                        mel_fake_v = (mel_real_v + delta_v).clamp_min(-8.0)
                        d_fake_v = D(mel_fake_v)
                        g_gan_v = -d_fake_v.mean()
                        g_spec_v = F.l1_loss(mel_fake_v, mel_real_v)
                        lossG_val = (g_gan_v * LAMBDA_GAN) + (g_spec_v * LAMBDA_SPEC)

                        # ASV metric pe VAL
                        val_evasion = np.nan
                        try:
                            wave_fake_v = mel2wav(mel_fake_v)
                            asv_vec_v = make_asv_vector_from_wave(wave_fake_v)
                            x_np_v = asv_vec_v.detach().cpu().numpy().astype("float32")
                            x_np_v = asv_scaler.transform(x_np_v)
                            preds_v = keras_model.predict(x_np_v, verbose=0)
                            if preds_v.ndim == 2:
                                if asv_head == "softmax":
                                    probs_v = softmax(preds_v, axis=1)
                                    p_target_v = probs_v[:, target_idx]
                                else:
                                    p_sig_v = preds_v[:, 0].astype("float32")
                                    p_target_v = p_sig_v if sigmoid_mapping == "target_is_pos" else (1.0 - p_sig_v)
                                val_evasion = float(p_target_v.mean())
                        except Exception:
                            val_evasion = np.nan

                        # log val
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([
                                int(time.time()), epoch, global_step, "val",
                                np.nan, float(lossG_val.cpu()), float(g_gan_v.cpu()), float(g_spec_v.cpu()),
                                np.nan, optG.param_groups[0]["lr"], optD.param_groups[0]["lr"],
                                delta_scale, inst_sigma, np.nan, val_evasion
                            ])

                        # samples audio (și pe pași)
                        samples_dir = SAVE_DIR / "samples" / f"step{global_step:07d}"
                        with torch.no_grad():
                            wave_fake_v = mel2wav(mel_fake_v)  # [B,1,T]
                            save_audio_batch(samples_dir, wave_fake_v, SR, max_items=4, tag="fake")
                            save_audio_batch(samples_dir, wave_val,    SR, max_items=4, tag="real")

                        # checkpoint pe val-step
                        save_checkpoint({
                            "epoch": epoch, "step": global_step,
                            "G": G.state_dict(), "D": D.state_dict(),
                            "optG": optG.state_dict(), "optD": optD.state_dict(),
                        }, name=f"step{global_step:07d}.pth")

                        # reducere LR pe platou (după lossG_val)
                        schedG.step(lossG_val)
                        schedD.step(lossG_val)

                        print(f"[VAL@step] step={global_step} epoch={epoch} "
                              f"lossG_val={lossG_val:.6f} g_spec={g_spec_v:.6f} "
                              f"val_evasion={val_evasion if not np.isnan(val_evasion) else float('nan'):.4f}")

                # programează următoarea validare
                next_val_step += VAL_INTERVAL

            # checkpointuri periodice ușoare (opțional)
            if (global_step % (LOG_INTERVAL * 10)) == 0 and global_step > 0:
                save_checkpoint({
                    "epoch": epoch, "step": global_step,
                    "G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                }, name=f"optim_step{global_step:07d}.pth")

            global_step += 1

        # --------- MOSTRE + METRICE LA FIECARE EPOCĂ ----------
        G.eval(); D.eval()

        # medii pe epocă
        ep_lossD = ep_lossD_sum / max(1, ep_steps)
        ep_lossG = ep_lossG_sum / max(1, ep_steps)
        ep_g_gan = ep_g_gan_sum / max(1, ep_steps)
        ep_g_spec = ep_g_spec_sum / max(1, ep_steps)
        ep_r1     = ep_r1_sum    / max(1, ep_steps)
        ep_evasion = (ep_evasion_sum / ep_evasion_cnt) if ep_evasion_cnt > 0 else np.nan

        # scriem linia „epoch” în CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()), epoch, global_step, "epoch",
                ep_lossD, ep_lossG, ep_g_gan, ep_g_spec, ep_r1,
                optG.param_groups[0]["lr"], optD.param_groups[0]["lr"],
                delta_scale, inst_sigma, ep_evasion, np.nan
            ])

        # mostre audio (always per-epoch)
        with torch.no_grad():
            # luăm un batch din val; dacă nu există, din train
            have_val = True
            try:
                batch_val = next(iter(val_loader))
            except StopIteration:
                have_val = False

            batch_src = batch_val if have_val else next(iter(train_loader))
            wave_src = batch_src["wave"].to(device)

            mel_real_ep = logmel_from_wave(wave_src)
            delta_ep = torch.tanh(G(mel_real_ep)) * delta_scale
            mel_fake_ep = (mel_real_ep + delta_ep).clamp_min(-8.0)
            wave_fake_ep = mel2wav(mel_fake_ep)
            samples_dir_ep = SAVE_DIR / "samples" / f"epoch{epoch:04d}"
            save_audio_batch(samples_dir_ep, wave_fake_ep, SR, max_items=4, tag="fake")
            save_audio_batch(samples_dir_ep, wave_src,     SR, max_items=4, tag="real")

        # --------- VALIDARE LA SFÂRȘIT DE EPOCĂ + BEST ---------
        lossG_val_epoch = None
        g_spec_v_epoch = None
        val_evasion_epoch = np.nan

        with torch.no_grad():
            val_ok = True
            try:
                batch_val = next(iter(val_loader))
            except StopIteration:
                val_ok = False

            if val_ok:
                wave_val = batch_val["wave"].to(device)
                mel_real_v = logmel_from_wave(wave_val)
                delta_v = torch.tanh(G(mel_real_v)) * delta_scale
                mel_fake_v = (mel_real_v + delta_v).clamp_min(-8.0)
                d_fake_v = D(mel_fake_v)
                g_gan_v = -d_fake_v.mean()
                g_spec_v = F.l1_loss(mel_fake_v, mel_real_v)
                lossG_val = (g_gan_v * LAMBDA_GAN) + (g_spec_v * LAMBDA_SPEC)

                # ASV metric
                try:
                    wave_fake_v = mel2wav(mel_fake_v)
                    asv_vec_v = make_asv_vector_from_wave(wave_fake_v)
                    x_np_v = asv_vec_v.detach().cpu().numpy().astype("float32")
                    x_np_v = asv_scaler.transform(x_np_v)
                    preds_v = keras_model.predict(x_np_v, verbose=0)
                    if preds_v.ndim == 2:
                        if asv_head == "softmax":
                            probs_v = softmax(preds_v, axis=1)
                            p_target_v = probs_v[:, target_idx]
                        else:
                            p_sig_v = preds_v[:, 0].astype("float32")
                            p_target_v = p_sig_v if sigmoid_mapping == "target_is_pos" else (1.0 - p_sig_v)
                        val_evasion_epoch = float(p_target_v.mean())
                except Exception:
                    val_evasion_epoch = np.nan

                lossG_val_epoch = float(lossG_val.cpu())
                g_spec_v_epoch = float(g_spec_v.cpu())

                # log „val_end”
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        int(time.time()), epoch, global_step, "val_end",
                        np.nan, lossG_val_epoch, float(g_gan_v.cpu()), g_spec_v_epoch,
                        np.nan, optG.param_groups[0]["lr"], optD.param_groups[0]["lr"],
                        delta_scale, inst_sigma, np.nan, val_evasion_epoch
                    ])

        # alegem metrica pentru „best”
        if lossG_val_epoch is not None:
            metric_val = g_spec_v_epoch if EARLY_STOP_METRIC == "val_spec" else lossG_val_epoch
        else:
            # fallback: fără val — folosim media pe epocă
            metric_val = ep_g_spec if EARLY_STOP_METRIC == "val_spec" else ep_lossG

        improved = metric_val < (best_val - 1e-6)
        if improved:
            best_val = metric_val
            best_epoch = epoch
            no_improve = 0
            save_checkpoint({
                "epoch": epoch, "step": global_step,
                "G": G.state_dict(), "D": D.state_dict(),
                "optG": optG.state_dict(), "optD": optD.state_dict(),
            }, name=f"best.pth")
        else:
            no_improve += 1

        # reducere LR pe platou pe baza lossG_val_epoch dacă există, altfel pe media epocii
        if lossG_val_epoch is not None:
            schedG.step(lossG_val_epoch)
            schedD.step(lossG_val_epoch)
            print(f"[VAL@end] epoch={epoch} lossG_val={lossG_val_epoch:.6f} best={best_val:.6f} @epoch {best_epoch} val_evasion={val_evasion_epoch if not np.isnan(val_evasion_epoch) else float('nan'):.4f}")
        else:
            schedG.step(ep_lossG)
            schedD.step(ep_lossG)
            print(f"[VAL@end] epoch={epoch} (no val) train_lossG={ep_lossG:.6f} best={best_val:.6f} @epoch {best_epoch}")

        # early stopping (numără epoci, nu doar validări pe pași)
        if EARLY_STOP_ENABLED and no_improve >= EARLY_STOP_PATIENCE:
            print(f"[EarlyStopping] Fără îmbunătățire {EARLY_STOP_PATIENCE} epoci. Oprim.")
            print("Training done.")
            return

    print("Training done.")

if __name__ == "__main__":
    main()
