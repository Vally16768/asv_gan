# train.py — DDP multi-GPU, GPU-only, microbatched GriffinLim, adaptive LR, early stopping, EMA-safe
from __future__ import annotations
import os, csv, time
from pathlib import Path
import numpy as np

# reduce CUDA fragmentation early
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
import torchaudio

# tqdm (rank0 only)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None
def tqdm_if_rank0(iterable, desc, rank0: bool):
    if rank0 and _tqdm is not None:
        return _tqdm(iterable, desc=desc, leave=False)
    return iterable

# TensorBoard (rank0 only)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ===== project imports
from constants import (
    # audio/feats
    SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    # training
    AMP_ENABLED, BATCH_SIZE, EPOCHS, CRITIC_ITERS,
    LR_G, LR_D, BETA1, BETA2,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_R1,
    LOG_INTERVAL, VAL_INTERVAL, SAVE_DIR, ROOT,
    # ASVspoof (placeholders, unused in-loop to keep GPU-only)
    ASV_MODEL_DIR, ASV_SCALER, ASV_COMBO, ASV_SR,
    EVASION_LAMBDA, EVASION_EVERY, TARGET_LABEL,
)
from utils import set_seed, EMA
from dataset import ASVBonafideDataset, pad_collate
from models import Generator, Critic
from losses import wgan_g_loss, wgan_d_loss, r1_regularizer
from features import stack_asv_features  # expects [B,1,T]

# ===== Optional Keras (not used in loop to keep GPU-only)
try:
    from detector_keras import load_keras_model
    HAVE_KERAS = True
except Exception:
    HAVE_KERAS = False
    def load_keras_model(*_a, **_k): return None

# ===== helpers
def spec_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)

class DeviceMel(nn.Module):
    """MelSpectrogram kept on the same device as input."""
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
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
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3 and wave.size(1) == 1:
            wave = wave.squeeze(1)
        self.mel = self.mel.to(wave.device)
        mel = self.mel(wave)
        mel = torch.log1p(mel.clamp(min=0.0))
        return _safe(mel)

class MelToWave(nn.Module):
    """InverseMel + GriffinLim on CUDA with micro-batching."""
    def __init__(self, gl_iters: int = 16, microbatch: int = 4):
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
            n_iter=gl_iters,   # lower to save mem/time
        )
        self.microbatch = max(1, int(microbatch))

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        dev = logmel.device
        self.inv_mel = self.inv_mel.to(dev)
        self.griffin = self.griffin.to(dev)

        B = logmel.size(0)
        outs = []
        # process small chunks on the same GPU to cap peak memory
        for i in range(0, B, self.microbatch):
            m = logmel[i:i + self.microbatch]
            mel = torch.expm1(m).clamp(min=0.0)
            spec = self.inv_mel(mel)      # [b, n_freq, T]
            wave = self.griffin(spec)     # [b, T]
            outs.append(wave)
            # free chunk temps
            del mel, spec, wave
            torch.cuda.synchronize()
        return torch.cat(outs, dim=0)  # [B, T]

def feats_from_wave(wave: torch.Tensor) -> torch.Tensor:
    # expects [B,1,T] on CUDA
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    return _safe(stack_asv_features(wave))

import soundfile as sf
def write_wav_np(wave_np: np.ndarray, sr: int, out_path: Path):
    if wave_np.ndim == 2:
        wave_np = wave_np[0]
    sf.write(str(out_path), wave_np, sr)
    return out_path

def _init_dirs():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "checkpoints").mkdir(exist_ok=True)
    (SAVE_DIR / "samples").mkdir(exist_ok=True)
    (SAVE_DIR / "tb").mkdir(exist_ok=True)

def _init_csv_logger(csv_path: Path, rank0: bool):
    if not rank0:
        return None, None
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

# ===== EMA wrappers (compatible)
def ema_update(ema_obj, model):
    try:
        ema_obj.update(model)
    except Exception:
        pass

def ema_get_model(ema_obj, fallback_model):
    for attr in ("ema", "module", "model"):
        if hasattr(ema_obj, attr):
            m = getattr(ema_obj, attr)
            if hasattr(m, "state_dict"):
                return m
    if hasattr(ema_obj, "state_dict"):
        return ema_obj
    return fallback_model

# ===== DDP utils
def setup_ddp():
    # Torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def cleanup_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

def all_reduce_bool(flag: bool) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor([1 if flag else 0], device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0

# ===== Main
def main():
    # perf niceties
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    is_ddp, rank, world_size, local_rank = setup_ddp()
    rank0 = (rank == 0)

    assert torch.cuda.is_available(), "CUDA required. No GPU detected."
    device = torch.device(f"cuda:{local_rank}")

    # Seed per-rank
    base_seed = 42
    set_seed(base_seed + rank)

    _init_dirs()
    csv_file, csv_writer = _init_csv_logger(SAVE_DIR / "metrics.csv", rank0=rank0)
    tb_writer = SummaryWriter(str(SAVE_DIR / "tb")) if (SummaryWriter is not None and rank0) else None

    # ===== Datasets & samplers
    train_ds = ASVBonafideDataset(split="train", use_validation=True)
    val_ds   = ASVBonafideDataset(split="val",   use_validation=True)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None

    # per-rank batch size to prevent global OOM
    per_rank_bs = max(1, BATCH_SIZE // world_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=per_rank_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_rank_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=pad_collate,
        persistent_workers=True
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS

    # ===== Infer channels (on this rank's data)
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        waves0 = first_batch["wave"].to(device).float()
        feats0 = first_batch["feats"]
        mel_mod = DeviceMel().to(device)
        mel0 = mel_mod(waves0)
    c_in_mel  = int(mel0.size(1))
    c_in_feat = int(feats0.size(1))

    # ===== Models
    G = Generator(c_in=c_in_mel).to(device)
    D = Critic(c_in=c_in_feat).to(device)

    # Wrap with DDP
    if is_ddp:
        G = DDP(G, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        D = DDP(D, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    # EMA tracks the unwrapped generator
    ema_target = G.module if isinstance(G, DDP) else G
    ema = EMA(ema_target, decay=0.999)

    # ===== Opt & sched (adaptive)
    optG = optim.Adam(ema_target.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(D.module.parameters() if isinstance(D, DDP) else D.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    schedG = optim.lr_scheduler.ReduceLROnPlateau(optG, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=rank0)
    schedD = optim.lr_scheduler.ReduceLROnPlateau(optD, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=rank0)

    scalerG = GradScaler(enabled=AMP_ENABLED)
    scalerD = GradScaler(enabled=AMP_ENABLED)

    # ===== Audio helpers on CUDA (microbatched GL)
    mel2wav = MelToWave(gl_iters=16, microbatch=max(1, per_rank_bs // 2 or 1)).to(device)
    mel_fn  = DeviceMel().to(device)
    tanh = nn.Tanh().to(device)
    delta_scale = 0.05

    # ===== Optional ASV (disabled in-loop to keep GPU-only)
    keras_model = None
    if HAVE_KERAS:
        try:
            keras_model = load_keras_model(ASV_MODEL_DIR / "best_model.keras")
        except Exception:
            try:
                keras_model = load_keras_model(ASV_MODEL_DIR / "best_model.h5")
            except Exception:
                keras_model = None

    # ===== Early stopping state (rank0 decides, then broadcast)
    best_lossG = float("inf")
    epochs_no_improve = 0
    EARLY_PATIENCE = 10

    if rank0:
        print(
            f"== ASV-GAN DDP Training ==\n"
            f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}\n"
            f"EPOCHS: {EPOCHS} | Global Batch: {BATCH_SIZE} | Per-rank Batch: {per_rank_bs} | Steps/epoch: {steps_per_epoch}\n"
            f"AMP: {AMP_ENABLED} | LR_G: {LR_G} | LR_D: {LR_D} | Critic iters: {CRITIC_ITERS}\n"
            f"λ_gan={LAMBDA_GAN} λ_spec={LAMBDA_SPEC} λ_R1={LAMBDA_R1}\n"
        )

    global_step = 0
    for epoch in range(EPOCHS):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        ema_target.train(); D.train()
        epoch_sums = dict(lossD=0.0, lossG=0.0, g_gan=0.0, g_spec=0.0, r1=0.0)
        t0 = time.time()

        pbar = tqdm_if_rank0(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", rank0=rank0)
        for batch in pbar:
            global_step += 1
            wave = _safe(batch["wave"].to(device).float())        # [B,1,T]
            feats_real = _safe(batch["feats"].to(device).float()) # [B,C,Tf]
            mel = mel_fn(wave)                                    # [B,M,Tf]

            # ---- Critic update
            with torch.no_grad():
                delta = tanh(ema_target(mel)) * delta_scale
                mel_fake = _safe(mel + delta)
                wave_fake = mel2wav(mel_fake)                     # [B,T] on CUDA, microbatched
                wave_fake = _safe(wave_fake.unsqueeze(1))         # [B,1,T]
                feats_fake = feats_from_wave(wave_fake)           # [B,C,Tf] on CUDA

            optD.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=AMP_ENABLED):
                d_real = D(feats_real)
                d_fake = D(feats_fake)
                lossD = wgan_d_loss(d_real, d_fake)

            # R1 (fp32 path)
            feats_r1 = feats_real.detach().requires_grad_(True)
            d_real_fp32 = D(feats_r1)
            r1 = r1_regularizer(d_real_fp32, feats_r1)
            lossD_total = lossD + LAMBDA_R1 * r1

            scalerD.scale(lossD_total).backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
            scalerD.step(optD); scalerD.update()

            # extra critic iters
            for _ in range(max(0, CRITIC_ITERS - 1)):
                with torch.no_grad():
                    delta = tanh(ema_target(mel)) * delta_scale
                    mel_fake = _safe(mel + delta)
                    wave_fake = mel2wav(mel_fake)
                    wave_fake = _safe(wave_fake.unsqueeze(1))
                    feats_fake = feats_from_wave(wave_fake)
                optD.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=AMP_ENABLED):
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

            # ---- Generator update (using ema_target)
            optG.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=AMP_ENABLED):
                delta = tanh(ema_target(mel)) * delta_scale
                mel_fake = _safe(mel + delta)
                wave_fake = mel2wav(mel_fake)
                wave_fake = _safe(wave_fake.unsqueeze(1))
                feats_fake = feats_from_wave(wave_fake)

                g_gan  = wgan_g_loss(D(feats_fake))
                g_spec = spec_l1(mel_fake, mel)
                lossG  = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec

            scalerG.scale(lossG).backward()
            torch.nn.utils.clip_grad_norm_(ema_target.parameters(), max_norm=5.0)
            scalerG.step(optG); scalerG.update()

            # EMA tracks the (unwrapped) generator weights
            ema_update(ema, ema_target)

            # ---- accumulate
            epoch_sums["lossD"] += float(lossD.detach())
            epoch_sums["lossG"] += float(lossG.detach())
            epoch_sums["g_gan"] += float(g_gan.detach())
            epoch_sums["g_spec"] += float(g_spec.detach())
            epoch_sums["r1"]    += float(r1.detach())

            # ---- logging (rank0)
            if rank0 and (global_step % LOG_INTERVAL == 0):
                lrG = optG.param_groups[0]["lr"]
                lrD = optD.param_groups[0]["lr"]
                print(f"[ep {epoch}] step {global_step:>6}/{total_steps}  "
                      f"D={float(lossD):.3f}  G={float(lossG):.3f}  "
                      f"g_gan={float(g_gan):.3f}  g_spec={float(g_spec):.3f}  r1={float(r1):.3f}  "
                      f"lrG={lrG:.2e} lrD={lrD:.2e}")
                if csv_writer:
                    now = int(time.time())
                    csv_writer.writerow([now, epoch, global_step, "train",
                                         float(lossD), float(lossG),
                                         float(g_gan), float(g_spec), float(r1),
                                         lrG, lrD])
                    csv_file.flush()
                if tb_writer:
                    _tb(tb_writer, "train/lossD", float(lossD), global_step)
                    _tb(tb_writer, "train/lossG", float(lossG), global_step)
                    _tb(tb_writer, "train/g_gan", float(g_gan), global_step)
                    _tb(tb_writer, "train/g_spec", float(g_spec), global_step)
                    _tb(tb_writer, "train/r1", float(r1), global_step)
                    _tb(tb_writer, "lr/G", lrG, global_step)
                    _tb(tb_writer, "lr/D", lrD, global_step)

            # ---- samples + checkpoint (rank0)
            if rank0 and (global_step % VAL_INTERVAL == 0):
                with torch.no_grad():
                    b = min(2, wave_fake.shape[0])
                    for i in range(b):
                        wav = wave_fake[i].detach().float().cpu().numpy().astype(np.float32)
                        outp = SAVE_DIR / "samples" / f"epoch_{epoch:03d}_step_{global_step:06d}_{i}.wav"
                        write_wav_np(wav, SR, outp)
                # checkpoint (EMA)
                ema_model = ema_get_model(ema, ema_target)
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "G": (ema_target.state_dict()),
                    "G_ema": ema_model.state_dict(),
                    "D": (D.module.state_dict() if isinstance(D, DDP) else D.state_dict()),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict(),
                }
                torch.save(ckpt, SAVE_DIR / "checkpoints" / f"ckpt_ep{epoch:03d}_st{global_step:06d}.pth")

        # ===== end epoch (average across ranks)
        n = max(1, steps_per_epoch)
        # reduce sums across processes
        for k in epoch_sums:
            t = torch.tensor([epoch_sums[k]], device=device, dtype=torch.float32)
            if is_ddp:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_sums[k] = (t.item() / (dist.get_world_size() if is_ddp else 1))

        ep_lossD = epoch_sums["lossD"] / n
        ep_lossG = epoch_sums["lossG"] / n
        ep_g_gan = epoch_sums["g_gan"] / n
        ep_g_spec= epoch_sums["g_spec"] / n
        ep_r1    = epoch_sums["r1"] / n
        dt = time.time() - t0

        if rank0:
            print(f"[EPOCH {epoch+1}/{EPOCHS}] "
                  f"D={ep_lossD:.3f} G={ep_lossG:.3f} g_gan={ep_g_gan:.3f} g_spec={ep_g_spec:.3f} r1={ep_r1:.3f} "
                  f"| {dt:.1f}s")
            if csv_writer:
                now = int(time.time())
                lrG = optG.param_groups[0]['lr']
                lrD = optD.param_groups[0]['lr']
                csv_writer.writerow([now, epoch, global_step, "epoch",
                                     ep_lossD, ep_lossG, ep_g_gan, ep_g_spec, ep_r1,
                                     lrG, lrD])
                csv_file.flush()
            if tb_writer:
                _tb(tb_writer, "epoch/lossD", ep_lossD, epoch+1)
                _tb(tb_writer, "epoch/lossG", ep_lossG, epoch+1)
                _tb(tb_writer, "epoch/g_gan", ep_g_gan, epoch+1)
                _tb(tb_writer, "epoch/g_spec", ep_g_spec, epoch+1)
                _tb(tb_writer, "epoch/r1", ep_r1, epoch+1)

        # schedulers (use G/D epoch losses)
        schedG.step(ep_lossG)
        schedD.step(ep_lossD)

        # early stopping (rank0 decides)
        stop_here_rank0 = False
        if rank0:
            if ep_lossG < best_lossG - 1e-4:
                best_lossG = ep_lossG
                epochs_no_improve = 0
                ema_model = ema_get_model(ema, ema_target)
                torch.save(ema_model.state_dict(), SAVE_DIR / "G_best.pth")
                print(f"✓ New best generator loss: {best_lossG:.4f} — saved G_best.pth")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 10:  # patience
                print(f"⚠ Early stopping after {epoch+1} epochs — no improvement for {epochs_no_improve} epochs.")
                stop_here_rank0 = True

        # broadcast stop flag
        if all_reduce_bool(stop_here_rank0):
            break

    # ===== save final EMA (rank0)
    if rank0:
        ema_model = ema_get_model(ema, ema_target)
        torch.save(ema_model.state_dict(), SAVE_DIR / "G_ema.pth")
        print(f"Done. Saved EMA generator to: {SAVE_DIR / 'G_ema.pth'}")

    if tb_writer is not None:
        tb_writer.close()
    if csv_file is not None:
        csv_file.close()

    cleanup_ddp(is_ddp)

if __name__ == "__main__":
    main()
