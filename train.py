# train.py — DDP stabil: seed identic pentru model, mixup sincronizat,
# static_graph=True, Keras/ASV doar pe rank-0, broadcast surrogate o dată/epocă.
from __future__ import annotations
import os, random, datetime
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
    TARGET_P_BONA, TARGET_WINDOW, MIN_STEPS_TO_CHECK, MAX_TRAIN_STEPS,
    LOG_WITH_KERAS
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


# ---------- helpers numeric-stability ----------
def isfinite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()

def safe_tensor(x: torch.Tensor) -> torch.Tensor:
    # taie valori extreme, elimină NaN/Inf
    return torch.nan_to_num(x.clamp(min=-10.0, max=10.0), nan=0.0, posinf=1e4, neginf=-1e4)

def norm_wave_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, T] — normalizează per-sample la zero-mean / unit-std și limitează amplitudinea.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.where(std < eps, torch.full_like(std, eps), std)
    xn = (x - mean) / std
    return xn.clamp(-3.0, 3.0)  # ținem valori rezonabile înainte de D/G

# ---------- seeding ----------
def seed_all(s: int):
    import numpy as _np
    random.seed(s); torch.manual_seed(s); _np.random.seed(s)

def seed_data_rng(s: int):
    # RNG separat pentru augmentări; poate fi diferit pe rank fără să afecteze modelul/DDP
    import numpy as _np
    rnd = random.Random(s)
    g = torch.Generator()
    g.manual_seed(s)
    _np.random.seed(s)
    return rnd, g


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
    base = (ROOT / "database" / "data").resolve()
    exts = ["flac", "wav", "mp3", "ogg", "m4a"]
    return [str(base / "**" / f"*.{e}") for e in exts]


def broadcast_bool(flag: bool, device, world_size: int) -> bool:
    t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    if world_size > 1:
        dist.broadcast(t, src=0)
    return bool(int(t.item()))


def broadcast_tensor(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.broadcast(t, src=0)
    return t


def main():
    # Env sigure (corectați variabila deprecated)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # opțional pt. un singur nod instabil de rețea:
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_PROTO", "SIMPLE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus,))
    else:
        main_worker(0, 1)


def main_worker(rank: int, world_size: int):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=3600)
        )

    # --- Seed identic pentru MODEL (IMPORTANT) ---
    seed_all(SEED)               # toți rank-ii același seed pentru inițializarea modelului
    data_rnd, torch_data_gen = seed_data_rng(SEED + rank)  # RNG separat pentru augmentări

    # I/O
    for p in (SAVE_DIR, CKPT_DIR, SAMPLES_DIR):
        os.makedirs(p, exist_ok=True)

    # ----- Data -----
    roots = build_roots_patterns()
    train_ds = ASVBonafideDataset(roots=roots)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=pad_collate, drop_last=True,
        generator=torch_data_gen
    )

    # ----- Models (create cu seed identic) -----
    G = Generator().to(device)
    D = MultiScaleCritic().to(device)
    mel_feat = LogMel().to(device)
    mrstft = MRSTFTLoss().to(device)
    ema = EMA(G, decay=EMA_DECAY) if (USE_EMA and is_main(rank)) else None

    surrogate = SurrogateDetector(mel_bins=mel_feat.mel.n_mels, hidden=2048).to(device) if USE_SURROGATE else None

    # Keras ASV doar pe rank-0
    detwrap = DetectorWrapper() if is_main(rank) else None

    # DDP wrap — static_graph=True pentru a opri rebuild-ul bucket-urilor în runtime
    if world_size > 1:
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[rank], output_device=rank, broadcast_buffers=False, static_graph=True
        )
        D = torch.nn.parallel.DistributedDataParallel(
            D, device_ids=[rank], output_device=rank, broadcast_buffers=False, static_graph=True
        )
        if surrogate is not None:
            surrogate = torch.nn.parallel.DistributedDataParallel(
                surrogate, device_ids=[rank], output_device=rank, broadcast_buffers=False, static_graph=True
            )

    # Optims + scalers
    optG = optim.AdamW(G.parameters(), lr=LR_G, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optD = optim.AdamW(D.parameters(), lr=LR_D, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    optS = optim.Adam(surrogate.parameters(), lr=SURROGATE_LR,
                      betas=(SURROGATE_BETA1, SURROGATE_BETA2)) if surrogate is not None else None
    scalerD = GradScaler(enabled=AMP_ENABLED)
    scalerG = GradScaler(enabled=AMP_ENABLED)

    global_step = 0
    inst_noise = INST_NOISE_INIT
    delta = DELTA_INIT
    best_metric = -1.0
    lambda_evasion = 0.0
    want_stop = False

    if is_main(rank):
        print(f"Params G: {count_params(G)/1e6:.2f}M, D: {count_params(D)/1e6:.2f}M, Surrogate: {(count_params(surrogate)/1e6 if surrogate else 0):.2f}M")

    p_window = deque(maxlen=TARGET_WINDOW)

    for epoch in range(1, EPOCHS + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = tqdm(total=len(train_loader), disable=not is_main(rank),
                    desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for it, (x, paths) in enumerate(train_loader):
            if want_stop:
                break

            G.train(); D.train()
            x = x.to(device, non_blocking=True)                 # [B, T] waveform
            x = norm_wave_per_sample(x)                         # *** stabilizare critică ***
            x = safe_tensor(x)

            # ----- MIXUP SINCRONIZAT -----
            want_mix = data_rnd.random() < 0.3
            want_mix = broadcast_bool(want_mix, device, world_size)
            if want_mix:
                # permutare hotărâtă pe rank-0 și broadcastată
                if is_main(rank):
                    perm = torch.randperm(x.size(0), device=device)
                else:
                    perm = torch.empty(x.size(0), device=device, dtype=torch.long)
                perm = broadcast_tensor(perm, world_size)
                x = 0.7 * x + 0.3 * x[perm]
                x = norm_wave_per_sample(x)                     # re-norm după mixup
                x = safe_tensor(x)

            # ----- forward G (tanh pentru a menține [-1,1])
            with autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), enabled=AMP_ENABLED):
                y_raw = G(x)                                    # [B, T]
                y = torch.tanh(y_raw)                           # *** principal anti-NaN ***
                y = safe_tensor(y)

            # ----- D train (WGAN + lazy R1)
            for _ in range(CRITIC_ITERS):
                optD.zero_grad(set_to_none=True)
                with autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), enabled=AMP_ENABLED):
                    xr = x.unsqueeze(1)
                    xf = y.detach().unsqueeze(1)
                    if inst_noise > 0.0:
                        xr = (xr + torch.randn_like(xr) * inst_noise).clamp(-1, 1)
                        xf = (xf + torch.randn_like(xf) * inst_noise).clamp(-1, 1)
                    # safe
                    xr = safe_tensor(xr); xf = safe_tensor(xf)
                    sr, fr = D(xr)
                    sf, ff = D(xf)
                    lossD = d_loss_wgan(sr, sf)

                # R1 doar periodic (fp32, fără autocast)
                r1 = torch.tensor(0.0, device=device)
                if (global_step % R1_EVERY) == 0:
                    xr_fp32 = x.unsqueeze(1).detach().requires_grad_(True)
                    sr_r1, _ = D(xr_fp32)
                    r1 = r1_penalty(xr_fp32, sr_r1).clamp(max=1e3)

                lossD_total = lossD + LAMBDA_R1 * r1

                # protecție anti-NaN
                if not isfinite_tensor(lossD_total):
                    # micșorăm puțin bruiajul & delta și sărim pasul D
                    inst_noise = max(INST_NOISE_MIN, inst_noise * 0.99)
                    delta = max(DELTA_MIN, delta * 0.99)
                    continue

                scalerD.scale(lossD_total).backward()
                # grad clip
                try:
                    scalerD.unscale_(optD)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP if GRAD_CLIP and GRAD_CLIP > 0 else 5.0)
                except Exception:
                    pass
                scalerD.step(optD); scalerD.update()

            # ----- G train
            optG.zero_grad(set_to_none=True)
            with autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), enabled=AMP_ENABLED):
                sr_fake, ff_fake = D(y.unsqueeze(1))
                lossG_gan = g_loss_wgan(sr_fake) * LAMBDA_GAN

                # spectrale
                loss_mrstft = mrstft(y, x)
                mel_x = safe(mel_feat(x)); mel_y = safe(mel_feat(y))
                loss_mel = torch.nn.functional.l1_loss(mel_y, mel_x)

                # feature matching
                with torch.no_grad():
                    sr_real, ff_real = D(x.unsqueeze(1))
                loss_fm = feature_matching_loss(ff_real, ff_fake) * LAMBDA_FM
                loss_spec = (loss_mrstft + loss_mel) * LAMBDA_SPEC

                # evasion schedule
                if global_step < EVASION_WARMUP_STEPS:
                    lambda_evasion = 0.0
                else:
                    t = min(1.0, (global_step - EVASION_WARMUP_STEPS) / max(1, EVASION_RAMP_STEPS))
                    lambda_evasion = float(LAMBDA_EVASION_MAX * t)

                loss_evasion = torch.tensor(0.0, device=device)
                if (surrogate is not None) and (lambda_evasion > 0):
                    logits_bona = surrogate(mel_y)
                    loss_evasion = evasion_loss_from_logits(logits_bona, weight=SURROGATE_W) * lambda_evasion

                lossG_total = lossG_gan + loss_spec + loss_fm + loss_evasion

            # protecție anti-NaN
            if not isfinite_tensor(lossG_total):
                inst_noise = max(INST_NOISE_MIN, inst_noise * 0.99)
                delta = max(DELTA_MIN, delta * 0.99)
                continue

            scalerG.scale(lossG_total).backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                try:
                    scalerG.unscale_(optG)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
                except Exception:
                    pass
            scalerG.step(optG); scalerG.update()

            if ema is not None:
                ema.update(G.module if hasattr(G, "module") else G)

            # ----- Surrogate update: DOAR rank-0 (rar)
            if is_main(rank) and (surrogate is not None) and (global_step % SURROGATE_UPDATE_EVERY == 0):
                with torch.no_grad():
                    mel_y_det = safe(mel_feat(y.detach()))
                try:
                    if detwrap is not None and hasattr(detwrap, "keras_prob"):
                        keras_np = detwrap.keras_prob(y.detach().cpu())
                        keras_t = torch.from_numpy(keras_np).float().to(device)
                        optS.zero_grad(set_to_none=True)
                        logits = surrogate(mel_y_det)
                        loss_sur = torch.nn.functional.binary_cross_entropy_with_logits(logits, keras_t)
                        loss_sur.backward()
                        optS.step()
                except Exception as e:
                    if is_main(rank):
                        print(f"[ASVmodel] Surrogate update skipped: {e}")

            # ----- Schedules
            inst_noise = max(INST_NOISE_MIN, inst_noise * INST_NOISE_DECAY)
            delta = max(DELTA_MIN, delta * DELTA_DECAY)

            # ----- Logging (rank-0)
            if is_main(rank) and (global_step % LOG_INTERVAL == 0):
                if LOG_WITH_KERAS and detwrap is not None and hasattr(detwrap, "keras_prob"):
                    try:
                        p_bona = detwrap.keras_prob(y.detach().cpu())
                        p_bona_mean = float(np.mean(p_bona))
                    except Exception:
                        p_bona_mean = 0.0
                else:
                    with torch.no_grad():
                        if surrogate is not None:
                            logits = surrogate(safe(mel_feat(y.detach())))
                            p_bona_mean = float(torch.sigmoid(logits).mean().item())
                        else:
                            p_bona_mean = 0.0

                # toate valorile din log sunt finite
                def fnum(v):
                    if isinstance(v, torch.Tensor):
                        v = float(v.detach().cpu().item())
                    if not np.isfinite(v):
                        return 0.0
                    return float(v)

                row = {
                    "step": int(global_step), "epoch": int(epoch),
                    "lossD": fnum(lossD_total),
                    "lossG": fnum(lossG_total),
                    "loss_gan": fnum(lossG_gan),
                    "loss_spec": fnum(loss_spec),
                    "loss_fm": fnum(loss_fm),
                    "loss_evasion": fnum(loss_evasion),
                    "inst_noise": float(inst_noise), "delta": float(delta),
                    "lambda_evasion": float(lambda_evasion),
                    "p_bona_mean": float(p_bona_mean),
                }
                append_csv_row(Path(LOG_CSV), row)
                p_window.append(p_bona_mean)

                if p_bona_mean > best_metric:
                    best_metric = p_bona_mean
                    state = {"G": ddp_state_dict(G), "D": ddp_state_dict(D), "step": global_step, "epoch": epoch}
                    if ema is not None:
                        state["G_EMA"] = ema.shadow
                    torch.save(state, Path(CKPT_DIR) / "best.pth")
                    tqdm.write(f"[{global_step}] New best p_bona_mean = {best_metric:.4f} -> saved best.pth")

                # Early-stop (doar rank-0): oprim la finalul epocii
                if (global_step >= MIN_STEPS_TO_CHECK) and (len(p_window) == TARGET_WINDOW):
                    if float(np.mean(p_window)) >= TARGET_P_BONA:
                        tqdm.write(f"EARLY STOP candidate @ step {global_step}")
                        want_stop = True

            global_step += 1
            if is_main(rank): pbar.update(1)
            if global_step >= MAX_TRAIN_STEPS:
                want_stop = True

        # ---- final epocă: checkpoint + samples + (opțional) broadcast surrogate
        if is_main(rank):
            if SAVE_AUDIO_EVERY_EPOCH:
                with torch.no_grad():
                    save_wave(Path(SAMPLES_DIR) / f"ep{epoch:03d}_fake.wav", y[0].detach().cpu(), SR)
                    save_wave(Path(SAMPLES_DIR) / f"ep{epoch:03d}_real.wav", x[0].detach().cpu(), SR)
            state = {"G": ddp_state_dict(G), "D": ddp_state_dict(D), "step": global_step, "epoch": epoch}
            if ema is not None:
                state["G_EMA"] = ema.shadow
            torch.save(state, Path(CKPT_DIR) / f"epoch{epoch:03d}.pth")

        if world_size > 1:
            dist.barrier()
            if surrogate is not None:
                mod = surrogate.module if hasattr(surrogate, "module") else surrogate
                for p in mod.state_dict().values():
                    dist.broadcast(p.data, src=0)
            dist.barrier()

        if is_main(rank): pbar.close()
        if want_stop:
            break

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
