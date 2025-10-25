# train.py — verbose progress + metrics CSV/TensorBoard + checkpoints
from __future__ import annotations
from pathlib import Path
import csv
import os
import time
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchaudio

# tqdm (fallback dacă nu e instalat)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # no-op fallback

# TensorBoard (opțional)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from constants import (
    # audio/feats
    SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    # training
    AMP_ENABLED, BATCH_SIZE, EPOCHS, CRITIC_ITERS,
    LR_G, LR_D, BETA1, BETA2,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_R1,
    LOG_INTERVAL, VAL_INTERVAL, SAVE_DIR, ROOT,
    # ASVspoof
    ASV_MODEL_DIR, ASV_SCALER, ASV_COMBO, ASV_SR,
    EVASION_LAMBDA, EVASION_EVERY, TARGET_LABEL,
)
from utils import set_seed, EMA
from dataset import ASVBonafideDataset, pad_collate
from models import Generator, Critic
from losses import wgan_g_loss, wgan_d_loss, r1_regularizer

# ===== Optional (dacă există fișierele); dacă nu, continuă fără evasion
try:
    from detector_keras import load_keras_model
    HAVE_KERAS = True
except Exception:
    HAVE_KERAS = False
    def load_keras_model(*_a, **_k): return None

# ---------- helpers ----------
def spec_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)

# ---------- Log-Mel (device-aware) & Griffin-Lim ----------
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

def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """
    Acceptă [B,1,T] / [B,T] și întoarce logmel [B,M,Tf].
    IMPORTANT: mutăm transformul pe același device ca intrarea (stft window!).
    """
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    mel = _mel.to(wave.device)(wave)              # <<< device-aware
    mel = torch.log1p(torch.clamp(mel, min=0.0))  # stabil numeric
    return _safe(mel)

class MelToWave(torch.nn.Module):
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
            n_iter=60,
        )
    @torch.no_grad()
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        mel = torch.expm1(logmel).clamp(min=0.0)
        spec = self.inv_mel(mel)
        wave = self.griffin(spec)
        return wave

# ---------- Feature stack (folosim pipeline-ul tău)
from features import stack_asv_features
def feats_from_wave(wave: torch.Tensor) -> torch.Tensor:
    # stack_asv_features așteaptă [B,1,T]
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    return _safe(stack_asv_features(wave))

# ---------- temp wav writer (dacă salvezi mostre) ----------
import soundfile as sf
def write_temp_wav(wave_np: np.ndarray, sr: int, out_path: Path):
    if wave_np.ndim == 2:
        wave_np = wave_np[0]
    sf.write(str(out_path), wave_np, sr)
    return out_path

def _init_dirs():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)
    (SAVE_DIR / "samples").mkdir(exist_ok=True)
    (SAVE_DIR / "plots").mkdir(exist_ok=True)

def _init_csv_logger(csv_path: Path):
    is_new = not csv_path.exists()
    f = csv_path.open("a", newline="")
    w = csv.writer(f)
    if is_new:
        w.writerow([
            "time", "epoch", "step", "phase",
            "lossD", "lossG", "g_gan", "g_spec", "r1",
            "lrG", "lrD"
        ])
        f.flush()
    return f, w

def _tb(writer, tag: str, scalar: float, global_step: int):
    if writer is not None:
        writer.add_scalar(tag, scalar, global_step=global_step)

torch.autograd.set_detect_anomaly(False)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _init_dirs()
    csv_file, csv_writer = _init_csv_logger(SAVE_DIR / "metrics.csv")
    tb_writer = SummaryWriter(str(SAVE_DIR / "tb")) if SummaryWriter is not None else None

    # Datasets & loaders
    train_ds = ASVBonafideDataset(split="train", use_validation=True)
    val_ds   = ASVBonafideDataset(split="val",   use_validation=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device=="cuda"),
        collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device=="cuda"),
        collate_fn=pad_collate
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS

    # Inferă canalele din date
    first_batch = next(iter(train_loader))
    feats0 = first_batch["feats"]
    waves0 = first_batch["wave"]
    with torch.no_grad():
        mel0 = logmel_from_wave(waves0.to(device))
    c_in_mel = int(mel0.size(1))      # = N_MELS
    c_in_feat = int(feats0.size(1))   # stiva de features

    # Modele
    G = Generator(c_in=c_in_mel).to(device)   # operează pe mel
    D = Critic(c_in=c_in_feat).to(device)     # critic pe stacked feats

    # EMA of G
    ema = EMA(G, decay=0.999)

    # Opt & sched
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    schedG = optim.lr_scheduler.ExponentialLR(optG, gamma=0.999)
    schedD = optim.lr_scheduler.ExponentialLR(optD, gamma=0.999)

    scalerG = GradScaler(enabled=(AMP_ENABLED and device=="cuda"))
    scalerD = GradScaler(enabled=(AMP_ENABLED and device=="cuda"))

    # ASVspoof (opțional; dacă lipsesc fișierele, trece mai departe)
    keras_model = None
    if HAVE_KERAS:
        try:
            keras_model = load_keras_model(ASV_MODEL_DIR / "best_model.keras")
        except Exception:
            try:
                keras_model = load_keras_model(ASV_MODEL_DIR / "best_model.h5")
            except Exception:
                keras_model = None

    mel2wav = MelToWave()
    tmp_dir = (ROOT / "tmp_asv"); tmp_dir.mkdir(exist_ok=True)

    labels = None
    labels_path = ASV_MODEL_DIR / "labels.txt"
    if labels_path.exists():
        labels = [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]
        target_idx = labels.index(TARGET_LABEL) if TARGET_LABEL in labels else 0
    else:
        target_idx = 0  # presupunem index 0 = bona_fide

    tanh = nn.Tanh()
    delta_scale = 0.05
    global_step = 0

    # ===== Info de start =====
    print(
        f"== ASV-GAN Training ==\n"
        f"Device: {device}\n"
        f"EPOCHS: {EPOCHS} | Batch: {BATCH_SIZE} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}\n"
        f"AMP: {AMP_ENABLED} | LR_G: {LR_G} | LR_D: {LR_D} | Critic iters: {CRITIC_ITERS}\n"
        f"λ_gan={LAMBDA_GAN} λ_spec={LAMBDA_SPEC} λ_R1={LAMBDA_R1}\n"
    )

    # ======= Bucla de antrenare (WGAN + R1) =======
    for epoch in range(EPOCHS):
        G.train(); D.train()
        epoch_sums = dict(lossD=0.0, lossG=0.0, g_gan=0.0, g_spec=0.0, r1=0.0)
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            global_step += 1
            wave = batch["wave"].to(device).float()        # [B,1,T]
            feats_real = batch["feats"].to(device).float() # [B,C,Tf]
            wave = _safe(wave); feats_real = _safe(feats_real)

            mel = logmel_from_wave(wave)                   # [B,M,Tf]

            # --- Critic update (cu fake detasat) ---
            with torch.no_grad():
                delta = tanh(G(mel)) * delta_scale
                mel_fake = _safe(mel + delta)
                wave_fake = mel2wav(mel_fake.cpu()).to(device)   # GL pe CPU -> back to device
                wave_fake = _safe(wave_fake.unsqueeze(1))        # [B,1,T]
                feats_fake = feats_from_wave(wave_fake)           # [B,C,Tf]

            optD.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(AMP_ENABLED and device=="cuda")):
                d_real = D(feats_real)
                d_fake = D(feats_fake)
                lossD = wgan_d_loss(d_real, d_fake)

            # R1 pe real (fp32)
            feats_r1 = feats_real.detach().requires_grad_(True)
            d_real_fp32 = D(feats_r1)
            r1 = r1_regularizer(d_real_fp32, feats_r1)
            lossD_total = lossD + LAMBDA_R1 * r1

            scalerD.scale(lossD_total).backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
            scalerD.step(optD); scalerD.update(); schedD.step()

            # (opțional) iterații suplimentare Critic
            for _ in range(max(0, CRITIC_ITERS - 1)):
                with torch.no_grad():
                    delta = tanh(G(mel)) * delta_scale
                    mel_fake = _safe(mel + delta)
                    wave_fake = mel2wav(mel_fake.cpu()).to(device)
                    wave_fake = _safe(wave_fake.unsqueeze(1))
                    feats_fake = feats_from_wave(wave_fake)

                optD.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=(AMP_ENABLED and device=="cuda")):
                    d_real = D(feats_real)
                    d_fake = D(feats_fake)
                    lossD = wgan_d_loss(d_real, d_fake)
                feats_r1 = feats_real.detach().requires_grad_(True)
                d_real_fp32 = D(feats_r1)
                r1 = r1_regularizer(d_real_fp32, feats_r1)
                lossD_total = lossD + LAMBDA_R1 * r1
                scalerD.scale(lossD_total).backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                scalerD.step(optD); scalerD.update()

            # --- Generator update ---
            optG.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(AMP_ENABLED and device=="cuda")):
                delta = tanh(G(mel)) * delta_scale
                mel_fake = _safe(mel + delta)
                wave_fake = mel2wav(mel_fake.cpu()).to(device)
                wave_fake = _safe(wave_fake.unsqueeze(1))
                feats_fake = feats_from_wave(wave_fake)

                g_gan  = wgan_g_loss(D(feats_fake))
                g_spec = spec_l1(mel_fake, mel)
                lossG  = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec

            # (opțional) evasion în ASVspoof (dacă modelul e disponibil)
            if keras_model is not None and EVASION_LAMBDA > 0.0 and (global_step % EVASION_EVERY) == 0:
                # TODO: integrează pipeline-ul tău pentru vectorul de intrare în keras_model
                pass

            scalerG.scale(lossG).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            scalerG.step(optG); scalerG.update(); schedG.step()
            ema.update(G)

            # ---- agregare & vizibilitate ----
            lrG = optG.param_groups[0]["lr"]
            lrD = optD.param_groups[0]["lr"]
            epoch_sums["lossD"] += float(lossD.detach())
            epoch_sums["lossG"] += float(lossG.detach())
            epoch_sums["g_gan"] += float(g_gan.detach())
            epoch_sums["g_spec"] += float(g_spec.detach())
            epoch_sums["r1"]    += float(r1.detach())

            # progress bar postfix
            pbar.set_postfix({
                "D": f"{float(lossD):.3f}",
                "G": f"{float(lossG):.3f}",
                "g_gan": f"{float(g_gan):.3f}",
                "g_spec": f"{float(g_spec):.3f}",
                "r1": f"{float(r1):.3f}"
            })

            # logging: consolă + CSV + TensorBoard
            if global_step % LOG_INTERVAL == 0:
                msg = (f"[ep {epoch}] step {global_step:>6}/{total_steps}  "
                       f"D={float(lossD):.3f}  G={float(lossG):.3f}  "
                       f"g_gan={float(g_gan):.3f}  g_spec={float(g_spec):.3f}  r1={float(r1):.3f}  "
                       f"lrG={lrG:.2e} lrD={lrD:.2e}")
                print(msg)

            now = int(time.time())
            csv_writer.writerow([now, epoch, global_step, "train",
                                 float(lossD), float(lossG),
                                 float(g_gan), float(g_spec), float(r1),
                                 lrG, lrD])
            csv_file.flush()
            _tb(tb_writer, "train/lossD", float(lossD), global_step)
            _tb(tb_writer, "train/lossG", float(lossG), global_step)
            _tb(tb_writer, "train/g_gan", float(g_gan), global_step)
            _tb(tb_writer, "train/g_spec", float(g_spec), global_step)
            _tb(tb_writer, "train/r1", float(r1), global_step)
            _tb(tb_writer, "lr/G", lrG, global_step)
            _tb(tb_writer, "lr/D", lrD, global_step)

            # mostre & checkpoint la interval
            if global_step % VAL_INTERVAL == 0:
                # salvează câteva wav-uri sintetizate
                with torch.no_grad():
                    b = min(2, wave_fake.shape[0])
                    for i in range(b):
                        wav = wave_fake[i].detach().cpu().numpy().astype(np.float32)
                        outp = SAVE_DIR / "samples" / f"epoch_{epoch:03d}_step_{global_step:06d}_{i}.wav"
                        write_temp_wav(wav, SR, outp)

                # checkpoint
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "G": G.state_dict(),
                    "G_ema": ema.ema.state_dict(),
                    "D": D.state_dict(),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict(),
                    "schedG": schedG.state_dict(),
                    "schedD": schedD.state_dict(),
                }
                torch.save(ckpt, SAVE_DIR / "checkpoints" / f"ckpt_ep{epoch:03d}_st{global_step:06d}.pth")

        # ---- summary pe epocă ----
        n = steps_per_epoch if steps_per_epoch > 0 else 1
        ep_lossD = epoch_sums["lossD"] / n
        ep_lossG = epoch_sums["lossG"] / n
        ep_g_gan = epoch_sums["g_gan"] / n
        ep_g_spec= epoch_sums["g_spec"] / n
        ep_r1    = epoch_sums["r1"] / n
        dt = time.time() - t0

        print(f"[EPOCH {epoch+1}/{EPOCHS}] "
              f"D={ep_lossD:.3f} G={ep_lossG:.3f} g_gan={ep_g_gan:.3f} g_spec={ep_g_spec:.3f} r1={ep_r1:.3f} "
              f"| {dt:.1f}s")

        now = int(time.time())
        csv_writer.writerow([now, epoch, global_step, "epoch",
                             ep_lossD, ep_lossG, ep_g_gan, ep_g_spec, ep_r1,
                             optG.param_groups[0]['lr'], optD.param_groups[0]['lr']])
        csv_file.flush()
        _tb(tb_writer, "epoch/lossD", ep_lossD, epoch+1)
        _tb(tb_writer, "epoch/lossG", ep_lossG, epoch+1)
        _tb(tb_writer, "epoch/g_gan", ep_g_gan, epoch+1)
        _tb(tb_writer, "epoch/g_spec", ep_g_spec, epoch+1)
        _tb(tb_writer, "epoch/r1", ep_r1, epoch+1)

    # salvează G (EMA) final
    torch.save(ema.ema.state_dict(), SAVE_DIR / "G_ema.pth")
    print(f"Training complete. Saved EMA generator to: {SAVE_DIR / 'G_ema.pth'}")

    if tb_writer is not None:
        tb_writer.close()
    csv_file.close()

if __name__ == "__main__":
    main()
