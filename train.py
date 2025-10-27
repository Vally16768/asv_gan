# train.py — tuned WGAN training loop with dual GradScalers (fix for CRITIC_ITERS>1),
# stronger critic, TTUR, lazy R1, surrogate updates, and robust dataset roots (glob by ext).
from __future__ import annotations
import os, random
from pathlib import Path
from collections import deque
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from constants import (
    ROOT, SAVE_DIR, CKPT_DIR, LOG_CSV, SAMPLES_DIR,
    AMP_ENABLED, BATCH_SIZE, EPOCHS, NUM_WORKERS, PIN_MEMORY,
    LR_G, LR_D, BETA1, BETA2, WEIGHT_DECAY, CRITIC_ITERS, GRAD_CLIP,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_FM, LAMBDA_R1, R1_EVERY,
    DELTA_INIT, DELTA_MIN, DELTA_DECAY,
    INST_NOISE_INIT, INST_NOISE_MIN, INST_NOISE_DECAY,
    EVASION_WARMUP_STEPS, EVASION_RAMP_STEPS, LAMBDA_EVASION_MAX,
    LOG_INTERVAL, SAVE_AUDIO_EVERY_EPOCH,
    SEED, SR, USE_EMA, EMA_DECAY,
    USE_SURROGATE, SURROGATE_LR, SURROGATE_BETA1, SURROGATE_BETA2, SURROGATE_W,
    SURROGATE_UPDATE_EVERY,
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


# ----------------- helpers -----------------
def set_seed(seed: int):
    import numpy as _np
    random.seed(seed); torch.manual_seed(seed); _np.random.seed(seed)

def is_main(rank: int) -> bool:
    return rank == 0

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

def ddp_state_dict(m: nn.Module):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()

def build_roots_patterns() -> list[str]:
    """Returnează pattern-uri recursive pentru extensii audio comune în database/data."""
    base = (ROOT / "database" / "data").resolve()
    exts = ["flac", "wav", "mp3", "ogg", "m4a"]
    return [str(base / "**" / f"*.{e}") for e in exts]


# ----------------- DDP launcher -----------------
def main():
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus,))
    else:
        main_worker(0, 1)


def main_worker(rank: int, world_size: int):
    set_seed(SEED + rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # I/O
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # ----------------- Data + sampler (FIXED ROOTS) -----------------
    roots = build_roots_patterns()
    train_ds = ASVBonafideDataset(roots=roots)  # :contentReference[oaicite:0]{index=0}

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=pad_collate, drop_last=True
    )

    # Models
    G = Generator().to(device)                     # :contentReference[oaicite:1]{index=1}
    D = MultiScaleCritic().to(device)              # :contentReference[oaicite:2]{index=2}
    mel_feat = LogMel().to(device)                 # :contentReference[oaicite:3]{index=3}
    mrstft = MRSTFTLoss().to(device)               # :contentReference[oaicite:4]{index=4}
    ema = EMA(G, decay=EMA_DECAY) if (USE_EMA and is_main(rank)) else None

    # Surrogate + Keras detector (TF/CPU via wrapper)
    surrogate = SurrogateDetector(mel_bins=mel_feat.mel.n_mels, hidden=2048).to(device) if USE_SURROGATE else None  # :contentReference[oaicite:5]{index=5}
    detwrap = DetectorWrapper()  # Keras ASV (CPU) :contentReference[oaicite:6]{index=6}

    # DDP wrap
    if world_size > 1:
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        if surrogate is not None:
            surrogate = torch.nn.parallel.DistributedDataParallel(surrogate, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Optims (TTUR)
    optG = optim.AdamW(G.parameters(), lr=LR_G, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optD = optim.AdamW(D.parameters(), lr=LR_D, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optS = optim.Adam(surrogate.parameters(), lr=SURROGATE_LR, betas=(SURROGATE_BETA1, SURROGATE_BETA2)) if surrogate is not None else None

    # Dual scalers: unul pentru D (multiple steps per iter), unul pentru G
    scalerD = GradScaler(enabled=AMP_ENABLED)
    scalerG = GradScaler(enabled=AMP_ENABLED)

    global_step = 0
    inst_noise = INST_NOISE_INIT
    delta = DELTA_INIT
    best_metric = -1.0
    lambda_evasion = 0.0  # default pentru logging

    if is_main(rank):
        print(f"Params G: {count_params(G)/1e6:.2f}M, D: {count_params(D)/1e6:.2f}M, Surrogate: {(count_params(surrogate)/1e6 if surrogate else 0):.2f}M")

    p_window = deque(maxlen=TARGET_WINDOW)
    should_stop = torch.tensor(0, device=device)

    for epoch in range(1, EPOCHS + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        iters = len(train_loader)
        pbar = tqdm(total=iters, disable=not is_main(rank), desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for it, (x, paths) in enumerate(train_loader):
            G.train(); D.train()
            x = x.to(device, non_blocking=True)  # [B,T]

            # ușor mixup pt. diversitate
            if random.random() < 0.3:
                perm = torch.randperm(x.size(0), device=device)
                x = 0.7 * x + 0.3 * x[perm]

            # -------- G forward
            with autocast(device_type='cuda', enabled=AMP_ENABLED):
                y = G(x)  # [B,T]

            # -------- D train (WGAN-R1) with Lazy R1 & scalerD
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
                    lossD = d_loss_wgan(sr, sf)  # :contentReference[oaicite:7]{index=7}

                # lazy R1 în FP32
                r1 = torch.tensor(0.0, device=device)
                if (global_step % R1_EVERY) == 0:
                    xr_fp32 = x.unsqueeze(1).detach().requires_grad_(True)
                    sr_r1, _ = D(xr_fp32)  # FP32
                    r1_val = r1_penalty(xr_fp32, sr_r1).clamp(max=1e3)  # :contentReference[oaicite:8]{index=8}
                    r1 = r1_val

                lossD_total = lossD + LAMBDA_R1 * r1

                scalerD.scale(lossD_total).backward()
                scalerD.step(optD)
                scalerD.update()

            # -------- G train (scalerG)
            optG.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=AMP_ENABLED):
                xf = y.unsqueeze(1)
                sr_fake, ff_fake = D(xf)
                lossG_gan = g_loss_wgan(sr_fake) * LAMBDA_GAN      # :contentReference[oaicite:9]{index=9}

                # ***** FIX OOM: RE-USE mrstft INSTANCE INSTEAD OF RE-CREATING IT *****
                loss_mrstft = mrstft(y, x)                        # :contentReference[oaicite:10]{index=10}
                mel_x = mel_feat(x)
                mel_y = mel_feat(y)
                loss_mel = torch.nn.functional.l1_loss(safe(mel_y), safe(mel_x))

                with torch.no_grad():
                    sr_real, ff_real = D(x.unsqueeze(1))
                loss_fm = feature_matching_loss(ff_real, ff_fake) * LAMBDA_FM  # :contentReference[oaicite:11]{index=11}
                loss_spec = (loss_mrstft + loss_mel) * LAMBDA_SPEC

                # schedule evasion
                if global_step < EVASION_WARMUP_STEPS:
                    lambda_evasion = 0.0
                else:
                    t = min(1.0, (global_step - EVASION_WARMUP_STEPS) / max(1, EVASION_RAMP_STEPS))
                    lambda_evasion = float(LAMBDA_EVASION_MAX * t)

                loss_evasion = torch.tensor(0.0, device=device)
                if (surrogate is not None) and (lambda_evasion > 0):
                    logits_bona = surrogate(mel_y)
                    loss_evasion = evasion_loss_from_logits(logits_bona, weight=SURROGATE_W) * lambda_evasion  # :contentReference[oaicite:12]{index=12}

                lossG_total = lossG_gan + loss_spec + loss_fm + loss_evasion

            scalerG.scale(lossG_total).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                scalerG.unscale_(optG)
                torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
            scalerG.step(optG)
            scalerG.update()

            # EMA doar pe rank-0
            if ema is not None:
                ema.update(G.module if hasattr(G, "module") else G)

            # -------- Surrogate training
            if (surrogate is not None) and (global_step % SURROGATE_UPDATE_EVERY == 0):
                with torch.no_grad():
                    mel_y_det = mel_feat(y.detach())
                try:
                    keras_np = detwrap.keras_prob(y.detach().cpu())    # :contentReference[oaicite:13]{index=13}
                    keras_t = torch.from_numpy(keras_np).float().to(device)
                    optS.zero_grad(set_to_none=True)
                    logits = surrogate(mel_y_det)
                    loss_sur = torch.nn.functional.binary_cross_entropy_with_logits(logits, keras_t)
                    loss_sur.backward()
                    optS.step()
                except Exception as e:
                    if is_main(rank):
                        print(f"[ASVmodel] Surrogate update skipped: {e}")

            # -------- Schedules
            inst_noise = max(INST_NOISE_MIN, inst_noise * INST_NOISE_DECAY)
            delta = max(DELTA_MIN, delta * DELTA_DECAY)

            # -------- logging + CSV + early-stop (doar rank-0 scrie/decide)
            if is_main(rank) and (global_step % LOG_INTERVAL == 0):
                with torch.no_grad():
                    p_bona = detwrap.keras_prob(y.detach().cpu())
                    p_bona_mean = float(np.mean(p_bona))

                row = {
                    "step": global_step, "epoch": epoch,
                    "lossD": float(lossD_total.detach().cpu().item()),
                    "lossG": float(lossG_total.detach().cpu().item()),
                    "loss_gan": float((lossG_gan.detach().cpu().item())),
                    "loss_spec": float((loss_spec.detach().cpu().item())),
                    "loss_fm": float((loss_fm.detach().cpu().item())),
                    "loss_evasion": float((loss_evasion.detach().cpu().item())) if isinstance(loss_evasion, torch.Tensor) else float(loss_evasion),
                    "inst_noise": float(inst_noise), "delta": float(delta),
                    "lambda_evasion": float(lambda_evasion),
                    "p_bona_mean": p_bona_mean,
                }
                append_csv_row(Path(LOG_CSV), row)
                p_window.append(p_bona_mean)

                if p_bona_mean > best_metric:
                    best_metric = p_bona_mean
                    state = {
                        "G": ddp_state_dict(G),
                        "D": ddp_state_dict(D),
                        "step": global_step, "epoch": epoch
                    }
                    if ema is not None:
                        state["G_EMA"] = ema.shadow
                    torch.save(state, Path(CKPT_DIR) / "best.pth")
                    tqdm.write(f"[{global_step}] New best p_bona_mean = {best_metric:.4f} -> saved best.pth")

                pbar.set_postfix({
                    "p_bona": f"{p_bona_mean:.3f}",
                    "lossG": f"{lossG_total.item():.3f}",
                    "lossD": f"{lossD_total.item():.3f}"
                })

                # early-stop
                if global_step >= MIN_STEPS_TO_CHECK and len(p_window) == TARGET_WINDOW:
                    rolling_mean = float(np.mean(p_window))
                    if rolling_mean >= TARGET_P_BONA:
                        tqdm.write(f"EARLY STOP: rolling_mean {rolling_mean:.4f} ≥ target {TARGET_P_BONA:.2f} @ step {global_step}")
                        should_stop.fill_(1)

            # răspândește decizia de early-stop tuturor proceselor
            if world_size > 1:
                dist.all_reduce(should_stop, op=dist.ReduceOp.SUM)
            if int(should_stop.item()) > 0:
                if is_main(rank):
                    final_state = {"G": ddp_state_dict(G), "D": ddp_state_dict(D), "step": global_step, "epoch": epoch}
                    if ema is not None:
                        final_state["G_EMA"] = ema.shadow
                    torch.save(final_state, Path(CKPT_DIR) / f"final_step{global_step:07d}.pth")
                if is_main(rank): pbar.close()
                return

            if global_step >= MAX_TRAIN_STEPS:
                if is_main(rank): pbar.close()
                return

            global_step += 1
            if is_main(rank): pbar.update(1)

        # end epoch
        if is_main(rank):
            if SAVE_AUDIO_EVERY_EPOCH:
                with torch.no_grad():
                    save_wave(Path(SAMPLES_DIR) / f"ep{epoch:03d}_fake.wav", y[0].detach().cpu(), SR)
                    save_wave(Path(SAMPLES_DIR) / f"ep{epoch:03d}_real.wav", x[0].detach().cpu(), SR)

            ckpt_path = Path(CKPT_DIR) / f"epoch{epoch:03d}.pth"
            state = {"G": ddp_state_dict(G), "D": ddp_state_dict(D), "step": global_step, "epoch": epoch}
            if ema is not None:
                state["G_EMA"] = ema.shadow
            torch.save(state, ckpt_path)
            pbar.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
