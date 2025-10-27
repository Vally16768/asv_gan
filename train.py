# train.py — early-stop la 80% bona_fide (rolling), autocast(device_type='cuda')
from __future__ import annotations
import os, time, math, random
from pathlib import Path
from collections import deque
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from constants import (
    ROOT, SAVE_DIR, CKPT_DIR, LOG_CSV, SAMPLES_DIR,
    AMP_ENABLED, BATCH_SIZE, EPOCHS, NUM_WORKERS, PIN_MEMORY,
    LR_G, LR_D, BETA1, BETA2, WEIGHT_DECAY, CRITIC_ITERS, GRAD_CLIP,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_FM, LAMBDA_R1,
    DELTA_INIT, DELTA_MIN, DELTA_DECAY,
    INST_NOISE_INIT, INST_NOISE_MIN, INST_NOISE_DECAY,
    EVASION_WARMUP_STEPS, EVASION_RAMP_STEPS, LAMBDA_EVASION_MAX,
    LOG_INTERVAL, SAVE_AUDIO_EVERY_EPOCH,
    SEED, SR, USE_EMA, EMA_DECAY,
    USE_SURROGATE, SURROGATE_LR, SURROGATE_BETA1, SURROGATE_BETA2, SURROGATE_W,
    TARGET_P_BONA, TARGET_WINDOW, MIN_STEPS_TO_CHECK, MAX_TRAIN_STEPS
)

from dataset import ASVBonafideDataset, pad_collate
from models import Generator, MultiScaleCritic, SurrogateDetector
from losses import (
    d_loss_wgan, g_loss_wgan, r1_penalty,
    MRSTFTLoss, feature_matching_loss, evasion_loss_from_logits
)
from features import LogMel, safe
from utils import append_csv_row, save_wave, count_params
from detector_wrapper import DetectorWrapper

torch.backends.cudnn.benchmark = True

def set_seed(seed):
    import numpy as _np
    random.seed(seed)
    torch.manual_seed(seed)
    _np.random.seed(seed)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Data
    train_ds = ASVBonafideDataset(roots=[str((ROOT / "database" / "data" / "**" / "*.flac").resolve())])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=pad_collate, drop_last=True)

    # Models
    G = Generator().to(device)
    D = MultiScaleCritic().to(device)
    mel_feat = LogMel().to(device)
    mrstft = MRSTFTLoss().to(device)

    ema = EMA(G, decay=EMA_DECAY) if USE_EMA else None

    # Surrogate & ASV detector (automat din ./ASVmodel)
    surrogate = SurrogateDetector(mel_bins=mel_feat.mel.n_mels, hidden=1024).to(device) if USE_SURROGATE else None
    detwrap = DetectorWrapper()  # auto-load Keras din ./ASVmodel (fără parametri)

    # Optims (TTUR)
    optG = optim.AdamW(G.parameters(), lr=LR_G, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optD = optim.AdamW(D.parameters(), lr=LR_D, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optS = None
    if surrogate is not None:
        optS = optim.Adam(surrogate.parameters(), lr=SURROGATE_LR, betas=(SURROGATE_BETA1, SURROGATE_BETA2))

    scaler = GradScaler(enabled=AMP_ENABLED)

    global_step = 0
    inst_noise = INST_NOISE_INIT
    delta = DELTA_INIT

    print(f"Params G: {count_params(G)/1e6:.2f}M, D: {count_params(D)/1e6:.2f}M, Surrogate: {(count_params(surrogate)/1e6 if surrogate else 0):.2f}M")

    # rolling window pentru media p_bona
    from collections import deque
    p_window = deque(maxlen=TARGET_WINDOW)
    best_metric = -1.0

    for epoch in range(1, EPOCHS + 1):
        for it, (x, paths) in enumerate(train_loader):
            G.train(); D.train()
            x = x.to(device)  # [B,T]

            # mixup ușor
            if random.random() < 0.3:
                perm = torch.randperm(x.size(0))
                x = 0.7 * x + 0.3 * x[perm]

            # ---- G forward
            with autocast(device_type='cuda', enabled=AMP_ENABLED):
                y = G(x)

            # ---- D train (WGAN-R1)
            for _ in range(CRITIC_ITERS):
                optD.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', enabled=AMP_ENABLED):
                    xr = x.unsqueeze(1)
                    xf = y.detach().unsqueeze(1)

                    if inst_noise > 0.0:
                        n_r = torch.randn_like(xr) * inst_noise
                        n_f = torch.randn_like(xf) * inst_noise
                        xr = (xr + n_r).clamp(-1, 1)
                        xf = (xf + n_f).clamp(-1, 1)

                    sr, fr = D(xr)
                    sf, ff = D(xf)
                    lossD = d_loss_wgan(sr, sf)

                    xr.requires_grad_(True)
                    sr_r1, _ = D(xr)
                    r1 = r1_penalty(xr, sr_r1).clamp(max=1e3)

                    lossD_total = lossD + LAMBDA_R1 * r1

                scaler.scale(lossD_total).backward()
                scaler.step(optD)

            # ---- G train
            optG.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=AMP_ENABLED):
                xf = y.unsqueeze(1)
                sr_fake, ff_fake = D(xf)
                lossG_gan = g_loss_wgan(sr_fake) * LAMBDA_GAN

                loss_mrstft = mrstft(y, x)
                mel_x = mel_feat(x)
                mel_y = mel_feat(y)
                loss_mel = torch.nn.functional.l1_loss(safe(mel_y), safe(mel_x))

                with torch.no_grad():
                    sr_real, ff_real = D(x.unsqueeze(1))
                loss_fm = feature_matching_loss(ff_real, ff_fake) * LAMBDA_FM
                loss_spec = (loss_mrstft + loss_mel) * LAMBDA_SPEC

                if global_step < EVASION_WARMUP_STEPS:
                    lambda_evasion = 0.0
                else:
                    t = min(1.0, (global_step - EVASION_WARMUP_STEPS) / max(1, EVASION_RAMP_STEPS))
                    lambda_evasion = LAMBDA_EVASION_MAX * t

                loss_evasion = torch.tensor(0.0, device=device)
                if surrogate is not None and lambda_evasion > 0:
                    logits_bona = surrogate(mel_y)
                    loss_evasion = evasion_loss_from_logits(logits_bona, weight=SURROGATE_W) * lambda_evasion

                lossG_total = lossG_gan + loss_spec + loss_fm + loss_evasion

            scaler.scale(lossG_total).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                scaler.unscale_(optG)
                torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
            scaler.step(optG)
            scaler.update()

            if USE_EMA:
                ema.update(G)

            # ---- Train surrogate împotriva Keras (fără grad la G)
            if surrogate is not None and (global_step % 5 == 0):
                with torch.no_grad():
                    mel_y_det = mel_feat(y.detach())
                try:
                    keras_np = detwrap.keras_prob(y.detach().cpu())
                    keras_t = torch.from_numpy(keras_np).float().to(device)
                    optS.zero_grad(set_to_none=True)
                    logits = surrogate(mel_y_det)
                    loss_sur = torch.nn.functional.binary_cross_entropy_with_logits(logits, keras_t)
                    loss_sur.backward()
                    optS.step()
                except Exception as e:
                    raise RuntimeError(f"[ASVmodel] Eroare la scorarea Keras: {e}")

            # ---- Schedules
            inst_noise = max(INST_NOISE_MIN, inst_noise * INST_NOISE_DECAY)
            delta = max(DELTA_MIN, delta * DELTA_DECAY)

            # ---- Logging
            if global_step % LOG_INTERVAL == 0:
                with torch.no_grad():
                    p_bona = detwrap.keras_prob(y.detach().cpu())
                    p_bona_mean = float(np.mean(p_bona))

                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "lossD": float(lossD_total.detach().cpu().item()) if 'lossD_total' in locals() else 0.0,
                    "lossG": float(lossG_total.detach().cpu().item()),
                    "loss_gan": float(lossG_gan.detach().cpu().item()),
                    "loss_spec": float(loss_spec.detach().cpu().item()),
                    "loss_fm": float(loss_fm.detach().cpu().item()),
                    "loss_evasion": float(loss_evasion.detach().cpu().item()) if isinstance(loss_evasion, torch.Tensor) else float(loss_evasion),
                    "inst_noise": float(inst_noise),
                    "delta": float(delta),
                    "lambda_evasion": float(lambda_evasion),
                    "p_bona_mean": p_bona_mean,
                }
                append_csv_row(Path(LOG_CSV), row)
                p_window.append(p_bona_mean)

                if p_bona_mean > best_metric:
                    best_metric = p_bona_mean
                    state = {"G": G.state_dict(), "D": D.state_dict(), "step": global_step, "epoch": epoch}
                    if USE_EMA:
                        state["G_EMA"] = ema.shadow
                    torch.save(state, Path(CKPT_DIR) / "best.pth")
                    print(f"[{global_step}] New best p_bona_mean = {best_metric:.4f} -> saved best.pth")

                print(f"step={global_step} epoch={epoch} it={it} p_bona_mean={p_bona_mean:.4f} lossG={lossG_total.item():.4f} lossD={lossD_total.item() if 'lossD_total' in locals() else 0.0:.4f}")

            global_step += 1

            # mostre rapide
            if SAVE_AUDIO_EVERY_EPOCH and (global_step % (LOG_INTERVAL * 10) == 0):
                with torch.no_grad():
                    wav_demo = y[0].detach().cpu()
                    save_wave(Path(SAMPLES_DIR) / f"step{global_step:07d}_fake.wav", wav_demo, SR)
                    save_wave(Path(SAMPLES_DIR) / f"step{global_step:07d}_real.wav", x[0].detach().cpu(), SR)

            # ---- Early stop (rolling mean ≥ 0.80)
            if global_step >= MIN_STEPS_TO_CHECK and len(p_window) == TARGET_WINDOW:
                rolling_mean = float(np.mean(p_window))
                if rolling_mean >= TARGET_P_BONA:
                    print(f"EARLY STOP: rolling_mean {rolling_mean:.4f} ≥ target {TARGET_P_BONA:.2f} la step {global_step}")
                    final_state = {"G": G.state_dict(), "D": D.state_dict(), "step": global_step, "epoch": epoch}
                    if USE_EMA:
                        final_state["G_EMA"] = ema.shadow
                    torch.save(final_state, Path(CKPT_DIR) / f"final_step{global_step:07d}.pth")
                    return

            if global_step >= MAX_TRAIN_STEPS:
                print("Reached MAX_TRAIN_STEPS. Exiting.")
                return

        # sfârșit epocă: salvează ckpt
        ckpt_path = Path(CKPT_DIR) / f"epoch{epoch:03d}.pth"
        state = {"G": G.state_dict(), "D": D.state_dict(), "step": global_step, "epoch": epoch}
        if USE_EMA:
            state["G_EMA"] = ema.shadow
        torch.save(state, ckpt_path)

    print("Training done. Best p_bona_mean =", best_metric)

if __name__ == "__main__":
    main()
