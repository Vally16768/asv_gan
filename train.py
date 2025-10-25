# train.py — stabilizat + best-only saves (checkpoints + samples) în checkpoints/, torchrun-safe
from __future__ import annotations
import os, csv, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchaudio

# ----------------- Imports proiect -----------------
from constants import (
    # audio/feats
    SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    FEATS_MEAN, FEATS_STD,
    # train
    AMP_ENABLED, BATCH_SIZE, EPOCHS, CRITIC_ITERS,
    LR_G, LR_D, BETA1, BETA2,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_R1,
    LOG_INTERVAL, VAL_INTERVAL, SAVE_DIR, ROOT,
    # schedules
    DELTA_INIT, DELTA_MIN, DELTA_DECAY,
    INST_NOISE_INIT, INST_NOISE_MIN, INST_NOISE_DECAY,
    # early stop
    EARLY_STOP_ENABLED, EARLY_STOP_PATIENCE, EARLY_STOP_METRIC,
    # ASV
    ASV_MODEL_DIR, ASV_SCALER, ASV_COMBO, ASV_SR,
    EVASION_LAMBDA, EVASION_EVERY, TARGET_LABEL,
)
from dataset import ASVBonafideDataset
from models import Generator, Critic
from losses import wgan_g_loss, wgan_d_loss, r1_regularizer
from utils import EMA

# ---------- rank helpers (torchrun-safe) ----------
def get_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0

def is_main_process() -> bool:
    return get_rank() == 0

# ---------- tqdm (opțional) ----------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

# ---------- TensorBoard (opțional) ----------
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# ============================================================
# ================  Features & Transformări  =================
# ============================================================

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
)

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """
    wave: [B, 1, T] float32 in [-1,1]
    return: log1p(mel) în [B, M, T']
    """
    mel = _mel.to(wave.device)(wave.squeeze(1))  # [B, M, T]
    mel = torch.log1p(mel).clamp(-8.0, 8.0)
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
            n_iter=80,
        )

    @torch.no_grad()
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        self.inv_mel = self.inv_mel.to(logmel.device)
        self.griffin = self.griffin.to(logmel.device)
        mel = torch.expm1(logmel).clamp(min=0.0)
        spec = self.inv_mel(mel)
        wave = self.griffin(spec)
        return wave

def _mfcc_from_logmel(logmel: torch.Tensor, n_mfcc: int = 20) -> torch.Tensor:
    B, M, T = logmel.shape
    device, dtype = logmel.device, logmel.dtype
    torch.manual_seed(0)
    W = torch.randn(M, n_mfcc, device=device, dtype=dtype) / np.sqrt(M)
    mfcc = torch.einsum("bmt,mk->bkt", logmel, W)
    return _safe(mfcc)

def _spectral_contrast(logmel: torch.Tensor, n_bands: int = 6) -> torch.Tensor:
    B, M, T = logmel.shape
    bands = []
    step = max(1, M // n_bands)
    for i in range(n_bands):
        s = i * step
        e = min(M, s + step)
        chunk = logmel[:, s:e, :]
        bands.append(chunk.max(dim=1, keepdim=True).values - chunk.min(dim=1, keepdim=True).values)
    return _safe(torch.cat(bands, dim=1))

def _temporal_stats(logmel: torch.Tensor) -> torch.Tensor:
    mu = logmel.mean(dim=1, keepdim=True)
    sd = logmel.std(dim=1, keepdim=True)
    d1 = torch.cat([torch.zeros_like(logmel[:, :, :1]), logmel[:, :, 1:] - logmel[:, :, :-1]], dim=2)
    d1 = d1.mean(dim=1, keepdim=True)
    return _safe(torch.cat([mu, sd, d1], dim=1))

def _wavelets_proxy(logmel: torch.Tensor) -> torch.Tensor:
    outs = []
    for s in (3, 5, 9):
        k = torch.ones(1, 1, s, device=logmel.device, dtype=logmel.dtype) / s
        y = torch.nn.functional.conv1d(logmel.reshape(-1, 1, logmel.shape[-1]), k, padding=s//2)
        outs.append(y.reshape_as(logmel))
    return _safe(torch.cat(outs, dim=1))

def _pitch_from_mel_proxy(logmel: torch.Tensor) -> torch.Tensor:
    B, M, T = logmel.shape
    freqs = torch.linspace(0.0, 1.0, M, device=logmel.device, dtype=logmel.dtype).view(1, M, 1)
    w = torch.exp(logmel).clamp_min(1e-8)
    centroid = (w * freqs).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
    return _safe(centroid)

def feats_from_mel_no_wave(logmel: torch.Tensor) -> torch.Tensor:
    B, M, T = logmel.shape
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)
    groups, gsize = 12, max(1, M // 12)
    chroma = []
    for i in range(groups):
        s = i * gsize; e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    chroma = _safe(torch.cat(chroma, dim=1))
    spec_contrast = _spectral_contrast(logmel, n_bands=6)
    temporal = _temporal_stats(logmel)
    pitch_like = _pitch_from_mel_proxy(logmel)
    wavelets = _wavelets_proxy(logmel)
    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch_like, wavelets], dim=1)
    feats = _safe(feats)
    feats = (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)
    return feats

try:
    from features import stack_asv_features as feats_from_wave
except Exception:
    def feats_from_wave(wave: torch.Tensor) -> torch.Tensor:
        logmel = logmel_from_wave(wave)
        return feats_from_mel_no_wave(logmel)

# ============================================================
# ================        ASVspoof wrapper      ===============
# ============================================================

def load_keras_safe() -> Optional[object]:
    try:
        from detector_keras import load_keras_model
        for name in ("best_model.keras", "best_model.h5"):
            p = ASV_MODEL_DIR / name
            if p.exists():
                return load_keras_model(p)
    except Exception:
        pass
    return None

def asv_evasion_penalty(keras_model, logmel_fake: torch.Tensor) -> torch.Tensor:
    try:
        feats = feats_from_mel_no_wave(logmel_fake)
        mu, sd = feats.mean(dim=-1), feats.std(dim=-1)
        pooled = torch.cat([mu, sd], dim=1)
        x_np = pooled.detach().cpu().numpy()
        preds = keras_model.predict(x_np, verbose=0).ravel()
        if not (np.all(preds >= 0.0) and np.all(preds <= 1.0)):
            from scipy.special import expit
            preds = expit(preds)
        loss_np = (1.0 - preds).mean()
        return torch.tensor(loss_np, dtype=logmel_fake.dtype, device=logmel_fake.device)
    except Exception:
        return torch.tensor(0.0, dtype=logmel_fake.dtype, device=logmel_fake.device)

# ============================================================
# ================        Utilitare train      ===============
# ============================================================

def add_instance_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return x
    return _safe(x + torch.randn_like(x) * std)

class EarlyStopper:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience

# ============================================================
# ================           main()            ===============
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- I/O dirs (rank 0) -----------------
    # Dacă SAVE_DIR e relativ -> ROOT/SAVE_DIR; dacă este absolut -> îl folosim ca atare.
    base_save = Path(SAVE_DIR)
    if not base_save.is_absolute():
        base_save = Path(ROOT) / base_save
    save_dir = base_save  # ar trebui să fie checkpoints/
    samples_dir = save_dir / "samples"
    tb_dir = save_dir / "tb"
    plots_dir = save_dir / "plots"
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Date -----------------
    train_set = ASVBonafideDataset(split="train")
    val_set   = ASVBonafideDataset(split="val")

    def _collate(batch):
        waves = [b["wave"] for b in batch]
        maxT = max(w.shape[-1] for w in waves)
        waves = [torch.nn.functional.pad(w, (0, maxT - w.shape[-1])) for w in waves]
        waves = torch.stack(waves, dim=0).float()  # [B,1,T]
        return {"wave": waves}

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=_collate, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=_collate, drop_last=False)

    # ----------------- Modele -----------------
    with torch.no_grad():
        tmp_wave = torch.zeros(2, 1, int(1.0 * SR))
        tmp_mel  = logmel_from_wave(tmp_wave)
        tmp_feats = feats_from_mel_no_wave(tmp_mel)
        C_feat = tmp_feats.shape[1]

    G = Generator(c_in=N_MELS).to(device)
    D = Critic(c_in=C_feat).to(device)
    ema = EMA(G, decay=0.999)

    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    scalerG = GradScaler(enabled=AMP_ENABLED)
    scalerD = GradScaler(enabled=AMP_ENABLED)

    spec_l1 = nn.L1Loss()

    # Schedules
    delta_scale = DELTA_INIT
    inst_noise_std = INST_NOISE_INIT
    def step_schedules():
        nonlocal delta_scale, inst_noise_std
        delta_scale = max(DELTA_MIN, delta_scale * DELTA_DECAY)
        inst_noise_std = max(INST_NOISE_MIN, inst_noise_std * INST_NOISE_DECAY)

    # ASVspoof
    keras_model = load_keras_safe()

    # TB + CSV (rank 0 only)
    tb_writer = None
    if is_main_process() and SummaryWriter is not None:
        tb_writer = SummaryWriter(str(tb_dir))
    csv_path = save_dir / "train_log.csv"
    csv_file = None
    csv_writer = None
    if is_main_process():
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time","epoch","step","D","G","g_gan","g_spec","r1","lrG","lrD","delta","inst_noise"])
        csv_file.flush()

    # Early stopping
    early = EarlyStopper(EARLY_STOP_PATIENCE, mode="min") if EARLY_STOP_ENABLED else None

    mel2wav = MelToWave()

    # ---------- checkpoint helpers (rank 0 only; best-only) ----------
    def save_ckpt_best(tag: str):
        if not is_main_process():
            return
        torch.save(ema.shadow.state_dict(), save_dir / f"G_ema_{tag}.pth")
        torch.save(G.state_dict(),             save_dir / f"G_train_{tag}.pth")
        torch.save(D.state_dict(),             save_dir / f"D_train_{tag}.pth")
        torch.save({
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "scalerG": scalerG.state_dict(),
            "scalerD": scalerD.state_dict(),
            "delta_scale": float(delta_scale),
            "inst_noise_std": float(inst_noise_std),
        }, save_dir / f"optim_{tag}.pth")

    best_val = float("inf")
    best_epoch = -1

    step_global = 0
    for epoch in range(EPOCHS):
        G.train(); D.train()
        ep_lossD = ep_lossG = ep_g_gan = ep_g_spec = ep_r1 = 0.0
        start_t = time.time()

        for ib, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            wave = batch["wave"].to(device)  # [B,1,T]
            mel  = logmel_from_wave(wave)    # [B,M,T]

            # ---------- Update D ----------
            for _ in range(CRITIC_ITERS):
                G.requires_grad_(False)
                D.requires_grad_(True)
                optD.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=AMP_ENABLED):
                    with torch.no_grad():
                        delta = torch.tanh(G(mel)) * delta_scale
                        mel_fake = _safe(mel + delta)

                    feats_real = feats_from_wave(wave)              # [B,C,T]
                    feats_real.requires_grad_(True)
                    feats_fake = feats_from_mel_no_wave(mel_fake)   # [B,C,T]

                    feats_real_noisy = add_instance_noise(feats_real, inst_noise_std)
                    feats_fake_noisy = add_instance_noise(feats_fake, inst_noise_std)

                    d_real = D(feats_real_noisy)
                    d_fake = D(feats_fake_noisy)
                    lossD = wgan_d_loss(d_real, d_fake)

                with torch.cuda.amp.autocast(enabled=False):
                    r1 = r1_regularizer(d_real.float(), feats_real_noisy.float()) * LAMBDA_R1
                    totalD = lossD + r1

                scalerD.scale(totalD).backward()
                scalerD.step(optD)
                scalerD.update()

                ep_lossD += float(lossD.detach().cpu())
                ep_r1    += float(r1.detach().cpu())
                step_schedules()
                step_global += 1

            # ---------- Update G ----------
            G.requires_grad_(True)
            D.requires_grad_(False)
            optG.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=AMP_ENABLED):
                delta = torch.tanh(G(mel)) * delta_scale
                mel_fake = _safe(mel + delta)
                feats_fake = feats_from_mel_no_wave(mel_fake)
                d_fake = D(add_instance_noise(feats_fake, inst_noise_std))
                g_gan = wgan_g_loss(d_fake)
                g_spec = spec_l1(mel_fake, mel)
                evasion = torch.tensor(0.0, device=device, dtype=mel.dtype)
                if keras_model is not None and EVASION_LAMBDA > 0.0 and (EVASION_EVERY <= 1 or (step_global % EVASION_EVERY) == 0):
                    evasion = asv_evasion_penalty(keras_model, mel_fake)
                lossG = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec + EVASION_LAMBDA * evasion

            scalerG.scale(lossG).backward()
            scalerG.step(optG)
            scalerG.update()
            ema.update(G)

            ep_lossG += float(lossG.detach().cpu())
            ep_g_gan += float(g_gan.detach().cpu())
            ep_g_spec += float(g_spec.detach().cpu())

            # ---------- Logging ----------
            if is_main_process() and (ib + 1) % LOG_INTERVAL == 0:
                lrG = optG.param_groups[0]["lr"]
                lrD = optD.param_groups[0]["lr"]
                print(f"[ep {epoch}] step {ib+1:4d}  D={ep_lossD/(ib+1):.3f}  G={ep_lossG/(ib+1):.3f}  g_gan={ep_g_gan/(ib+1):.3f}  g_spec={ep_g_spec/(ib+1):.3f}  r1={ep_r1/(ib+1):.3f}  Δ={delta_scale:.4f}  σ={inst_noise_std:.3f}")
                if csv_writer is not None:
                    csv_writer.writerow([int(time.time()), epoch, ib+1,
                                         f"{ep_lossD/(ib+1):.6f}", f"{ep_lossG/(ib+1):.6f}",
                                         f"{ep_g_gan/(ib+1):.6f}", f"{ep_g_spec/(ib+1):.6f}",
                                         f"{ep_r1/(ib+1):.6f}",
                                         f"{lrG:.6f}", f"{lrD:.6f}",
                                         f"{delta_scale:.6f}", f"{inst_noise_std:.6f}"])
                    csv_file.flush()  # type: ignore
                if tb_writer is not None:
                    tb_writer.add_scalar("train/D", ep_lossD/(ib+1), step_global)
                    tb_writer.add_scalar("train/G", ep_lossG/(ib+1), step_global)
                    tb_writer.add_scalar("train/g_gan", ep_g_gan/(ib+1), step_global)
                    tb_writer.add_scalar("train/g_spec", ep_g_spec/(ib+1), step_global)
                    tb_writer.add_scalar("train/r1", ep_r1/(ib+1), step_global)
                    tb_writer.add_scalar("train/delta_scale", delta_scale, step_global)
                    tb_writer.add_scalar("train/inst_noise", inst_noise_std, step_global)

            # FĂRĂ samples/checkpoints în timpul epocii — păstrăm doar best pe epocă

        # ----------------- VALIDARE PE EPOCĂ -----------------
        G.eval()
        val_spec_sum, val_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vw = vb["wave"].to(device).float()
                vm = logmel_from_wave(vw)
                delta = torch.tanh(ema.shadow(vm)) * delta_scale
                vm_fake = _safe(vm + delta)
                val_spec_sum += float(nn.functional.l1_loss(vm_fake, vm).detach().cpu())
                val_n += 1
        val_spec = val_spec_sum / max(1, val_n)
        dur = time.time() - start_t

        if is_main_process():
            print(f"[VAL] epoch {epoch+1}/{EPOCHS}  val_spec={val_spec:.4f}  time={dur/60.0:.1f}min")
            if tb_writer is not None:
                tb_writer.add_scalar("val/spec_l1", val_spec, epoch+1)
                tb_writer.add_scalar("epoch/delta_scale", delta_scale, epoch+1)
                tb_writer.add_scalar("epoch/inst_noise", inst_noise_std, epoch+1)
                tb_writer.add_scalar("epoch/lrG", optG.param_groups[0]['lr'], epoch+1)
                tb_writer.add_scalar("epoch/lrD", optD.param_groups[0]['lr'], epoch+1)
                tb_writer.add_scalar("epoch/g_gan", ep_g_gan / max(1, ib+1), epoch+1)
                tb_writer.add_scalar("epoch/g_spec", ep_g_spec / max(1, ib+1), epoch+1)
                tb_writer.add_scalar("epoch/r1", ep_r1 / max(1, ib+1), epoch+1)

        # ---------- BEST-ONLY: salvează doar dacă s-a îmbunătățit ----------
        improved = val_spec < best_val
        if improved:
            best_val = val_spec
            best_epoch = epoch + 1

            # Checkpoint best
            save_ckpt_best("best")

            # Samples best (2 exemple din batch de validare curent folosit mai sus)
            if is_main_process():
                with torch.no_grad():
                    # Folosim primele două wave-uri din primul batch de validare reîncărcat rapid
                    try:
                        vb0 = next(iter(val_loader))
                        wave_dbg = vb0["wave"][:2].to(device)
                    except Exception:
                        wave_dbg = torch.randn(2, 1, int(1.0 * SR), device=device)
                    mel_dbg = logmel_from_wave(wave_dbg)
                    delta_dbg = torch.tanh(ema.shadow(mel_dbg)) * delta_scale
                    mel_fake_dbg = _safe(mel_dbg + delta_dbg)
                    wav_fake = mel2wav(mel_fake_dbg)
                    for i in range(min(wav_fake.size(0), 2)):
                        wf = wav_fake[i].detach().cpu().unsqueeze(0).clamp_(-1.0, 1.0)
                        torchaudio.save(str(samples_dir / f"best_ep{best_epoch:03d}_{i}.wav"), wf, SR)
                print(f"[BEST] ep={best_epoch}  val_spec={best_val:.4f}  -> checkpoint + samples salvate în {save_dir}")

        # ---------- Early stop (nu mai salvează nimic în afară de best) ----------
        if EARLY_STOP_ENABLED and early is not None:
            stop = early.step(val_spec)
            if stop:
                if is_main_process():
                    print(f"[EarlyStop] Stop la epoca {epoch+1}. Best val_spec={early.best:.4f} @ epoca {best_epoch}")
                break

    # --------- Post-run: plot & închideri (nu salvează alte ckpt-uri) ----------
    if is_main_process():
        print(f"Training încheiat. Best @ epoca {best_epoch} cu val_spec={best_val:.4f}. Checkpoints & samples în: {save_dir}")

        # Auto-plot din CSV în PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(csv_path)
            x = np.arange(len(df))

            def _plot_one(ycols, title, fname):
                plt.figure()
                for col in ycols:
                    if col in df.columns:
                        plt.plot(x, df[col], label=col)
                plt.title(title)
                plt.xlabel("log step")
                plt.ylabel(title)
                plt.legend()
                plt.tight_layout()
                out = (save_dir / "plots" / fname)
                plt.savefig(out)
                plt.close()
                print(f"[PLOT] {out}")

            _plot_one(["D","G"], "Loss D/G", "loss_DG.png")
            _plot_one(["g_gan","g_spec","r1"], "GAN/Spec/R1", "components.png")
            _plot_one(["delta","inst_noise"], "Schedules", "schedules.png")
            _plot_one(["lrG","lrD"], "Learning Rates", "lrs.png")

        except Exception as e:
            print(f"[WARN] Plotting a eșuat: {e}")

        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
        except Exception:
            pass
        try:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
