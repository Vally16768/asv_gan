# train.py — full fixed with ASVspoof black-box evasion
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import torchaudio

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

# ===== Optional: robust Keras loader & ASV adapter
from detector_keras import load_keras_model
from asv_adapter import build_keras_input_vector

# ---------- small helpers ----------
def spec_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)

# ---------- Log-Mel and Griffin-Lim for reconstruction ----------
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
    # wave: [B,1,T] -> logmel: [B,M,Tf]  (log1p for stability)
    mel = _mel(wave)
    mel = torch.log1p(torch.clamp(mel, min=0.0))
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
        # logmel: [B,M,T] -> wave: [B,T]
        mel = torch.expm1(logmel).clamp(min=0.0)
        spec = self.inv_mel(mel)
        wave = self.griffin(spec)
        return wave

# ---------- Feature stack (proxy) ----------
# We will reuse the dataset-provided 'feats' for real; for fake we recompute
# by running the reconstructed wave through the dataset's feature pipeline.
# To avoid re-implementing stack_asv_features here, we import it:
from features import stack_asv_features
def feats_from_wave(wave: torch.Tensor) -> torch.Tensor:
    # stack_asv_features expects [B,1,T]
    return _safe(stack_asv_features(wave))

# ---------- temp wav writer ----------
import soundfile as sf
def write_temp_wav(wave_np: np.ndarray, sr: int, out_path: Path):
    if wave_np.ndim == 2:
        wave_np = wave_np[0]
    sf.write(str(out_path), wave_np, sr)
    return out_path

torch.autograd.set_detect_anomaly(False)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Peek first batch to infer channel counts
    first_batch = next(iter(train_loader))
    feats0 = first_batch["feats"]
    waves0 = first_batch["wave"]
    with torch.no_grad():
        mel0 = logmel_from_wave(waves0)
    c_in_mel = int(mel0.size(1))      # = N_MELS
    c_in_feat = int(feats0.size(1))   # stacked features channels

    # Models
    G = Generator(c_in=c_in_mel).to(device)   # operates on mel
    D = Critic(c_in=c_in_feat).to(device)     # critic on stacked feats

    # EMA of G
    ema = EMA(G, decay=0.999)

    # Opt & sched
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    schedG = optim.lr_scheduler.ExponentialLR(optG, gamma=0.999)
    schedD = optim.lr_scheduler.ExponentialLR(optD, gamma=0.999)

    scalerG = GradScaler(enabled=(AMP_ENABLED and device=="cuda"))
    scalerD = GradScaler(enabled=(AMP_ENABLED and device=="cuda"))

    # ASVspoof bits
    keras_model = load_keras_model(ASV_MODEL_DIR / "best_model.keras")
    mel2wav = MelToWave()
    tmp_dir = (ROOT / "tmp_asv"); tmp_dir.mkdir(exist_ok=True)

    labels = None
    labels_path = ASV_MODEL_DIR / "labels.txt"
    if labels_path.exists():
        labels = [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]
        target_idx = labels.index(TARGET_LABEL) if TARGET_LABEL in labels else 0
    else:
        target_idx = 0  # assume index 0 = bona_fide

    tanh = nn.Tanh()
    delta_scale = 0.05
    step = 0

    for ep in range(EPOCHS):
        G.train(); D.train()
        for batch in train_loader:
            step += 1
            # Inputs
            wave = batch["wave"].to(device).float()      # [B,1,T]
            feats_real = batch["feats"].to(device).float()   # [B,C,Tf]
            wave = _safe(wave); feats_real = _safe(feats_real)

            # Mel for generator
            mel = logmel_from_wave(wave)                 # [B,M,Tm]

            # ====== Update D (WGAN + R1) ======
            # Make a detached fake for D
            with torch.no_grad():
                delta_mel = tanh(G(mel)) * delta_scale
                mel_fake = _safe(mel + delta_mel)
                # reconstruct wave from mel_fake
                wave_fake = mel2wav(mel_fake.cpu()).to(device)  # [B,T]
                wave_fake = wave_fake.unsqueeze(1)              # [B,1,T]
                wave_fake = _safe(wave_fake)
                feats_fake = feats_from_wave(wave_fake)         # [B,C,Tf]

            # 1) main D step
            optD.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(AMP_ENABLED and device=="cuda")):
                d_real = D(feats_real)
                d_fake = D(feats_fake)
                lossD = wgan_d_loss(d_real, d_fake)
            # R1 on real (fp32)
            feats_r1 = feats_real.detach().requires_grad_(True)
            d_real_fp32 = D(feats_r1)
            r1 = r1_regularizer(d_real_fp32, feats_r1)
            lossD_total = lossD + LAMBDA_R1 * r1

            scalerD.scale(lossD_total).backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
            scalerD.step(optD); scalerD.update(); schedD.step()

            # 2) extra critic iters
            for _ in range(max(0, CRITIC_ITERS - 1)):
                with torch.no_grad():
                    delta_mel = tanh(G(mel)) * delta_scale
                    mel_fake = _safe(mel + delta_mel)
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
                scalerD.step(optD); scalerD.update(); schedD.step()

            # ====== Update G ======
            optG.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(AMP_ENABLED and device=="cuda")):
                delta_mel = tanh(G(mel)) * delta_scale
                mel_fake = _safe(mel + delta_mel)
                wave_fake = mel2wav(mel_fake.cpu()).to(device)
                wave_fake = _safe(wave_fake.unsqueeze(1))
                feats_fake = feats_from_wave(wave_fake)

                g_gan  = wgan_g_loss(D(feats_fake))
                g_spec = spec_l1(feats_fake, feats_real)
                lossG  = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec

            # --- ASVspoof evasion loss (black-box) ---
            if (step % EVASION_EVERY) == 0 and EVASION_LAMBDA > 0.0:
                with torch.no_grad():
                    wav_fake_np = wave_fake.detach().cpu().numpy().astype(np.float32)  # [B,1,T]
                    L_vals = []
                    subB = min(2, wav_fake_np.shape[0])
                    for i in range(subB):
                        wav_path = tmp_dir / f"fake_{ep}_{step}_{i}.wav"
                        write_temp_wav(wav_fake_np[i], ASV_SR, wav_path)
                        X = build_keras_input_vector(wav_path, ASV_COMBO, ASV_SR, ASV_SCALER)
                        logits = keras_model.predict(X, verbose=0)
                        if logits.ndim == 1:
                            logits = logits[None, :]
                        z = logits - np.max(logits, axis=1, keepdims=True)
                        p = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
                        p_bona = float(p[0, target_idx])  # maximize bona_fide prob
                        L_vals.append(-math.log(max(1e-6, p_bona)))
                    if L_vals:
                        L_evasion = torch.tensor(L_vals, device=device, dtype=torch.float32).mean()
                        lossG = lossG + EVASION_LAMBDA * L_evasion

            scalerG.scale(lossG).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            scalerG.step(optG); scalerG.update(); schedG.step()
            ema.update(G)

            if step % LOG_INTERVAL == 0:
                print(f"[ep {ep}] step {step}  D={lossD.item():.3f}  G={lossG.item():.3f}  "
                      f"g_gan={g_gan.item():.3f}  g_spec={g_spec.item():.3f}  r1={r1.item():.3f}")

        # -------- Validation --------
        if (ep % VAL_INTERVAL) == 0:
            total = 0.0; n = 0
            G.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    wave = batch["wave"].to(device).float()
                    feats_real = batch["feats"].to(device).float()
                    wave = _safe(wave); feats_real = _safe(feats_real)
                    mel = logmel_from_wave(wave)
                    delta_mel = nn.Tanh()(G(mel)) * 0.05
                    mel_fake = _safe(mel + delta_mel)
                    wave_fake = MelToWave()(mel_fake.cpu()).to(device)
                    wave_fake = _safe(wave_fake.unsqueeze(1))
                    feats_fake = feats_from_wave(wave_fake)
                    total += spec_l1(feats_fake, feats_real).item()
                    n += 1
            print(f"[val] epoch={ep} spec_l1={total/max(1,n):.4f}")

        # -------- Checkpoint EMA --------
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = {"state_dict": ema.shadow.state_dict(), "c_in_mel": c_in_mel, "c_in_feat": c_in_feat}
        torch.save(ckpt, SAVE_DIR / f"G_ema_ep{ep:03d}.pt")

if __name__ == "__main__":
    main()
