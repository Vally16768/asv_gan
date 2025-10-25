# train.py
from __future__ import annotations
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from constants import (
    AMP_ENABLED, BATCH_SIZE, EPOCHS, CRITIC_ITERS,
    LR_G, LR_D, BETA1, BETA2,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_R1,
    LOG_INTERVAL, VAL_INTERVAL, SAVE_DIR
)
from utils import set_seed, EMA
from dataset import ASVBonafideDataset, pad_collate
from models import Generator, Critic
from losses import wgan_g_loss, wgan_d_loss, r1_regularizer

def spec_l1(x, y):
    return (x - y).abs().mean()

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ASVBonafideDataset(split="train", use_validation=True)
    val_ds   = ASVBonafideDataset(split="val",   use_validation=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=pad_collate
    )

    sample = train_ds[0]["feats"]
    c_in = sample.size(0)

    G = Generator(c_in=c_in).to(device)
    D = Critic(c_in=c_in).to(device)
    ema = EMA(G, decay=0.999)

    optG = optim.AdamW(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.AdamW(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    total_steps = EPOCHS * max(1, len(train_loader))
    schedG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=total_steps)
    schedD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=total_steps)
    did_stepG = False
    did_stepD = False

    scalerG = GradScaler(enabled=AMP_ENABLED)
    scalerD = GradScaler(enabled=AMP_ENABLED)

    # limitator simplu pentru delta la început (prevenim explozii)
    delta_scale = 0.1
    tanh = nn.Tanh()

    step = 0
    for ep in range(EPOCHS):
        G.train(); D.train()
        for batch in train_loader:
            step += 1
            feats = batch["feats"].to(device)
            feats = torch.nan_to_num(feats, 0.0, 1e6, -1e6)

            # ====== D ======
            optD.zero_grad(set_to_none=True)

            # gen fake pentru D (sub autocast)
            with autocast(device_type="cuda", enabled=AMP_ENABLED and device == "cuda"):
                with torch.no_grad():
                    delta = tanh(G(feats)) * delta_scale
                    feats_fake = feats + delta

                d_real_amp = D(feats)        # pentru loss D (amp)
                d_fake_amp = D(feats_fake)
                lossD = wgan_d_loss(d_real_amp, d_fake_amp)

            # R1 în FP32, fără autocast: recomputăm D(feats) cu requires_grad=True
            feats_r1 = feats.detach().requires_grad_(True)
            d_real_fp32 = D(feats_r1)  # fp32
            r1 = r1_regularizer(d_real_fp32, feats_r1)
            lossD_total = lossD + LAMBDA_R1 * r1

            scalerD.scale(lossD_total).backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
            scalerD.step(optD)
            scalerD.update()
            did_stepD = True
            if did_stepD:
                schedD.step()

            # extra iterații Critic
            for _ in range(CRITIC_ITERS - 1):
                optD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    delta = tanh(G(feats)) * delta_scale
                    feats_fake = feats + delta
                with autocast(device_type="cuda", enabled=AMP_ENABLED and device == "cuda"):
                    d_real_amp = D(feats)
                    d_fake_amp = D(feats_fake)
                    lossD = wgan_d_loss(d_real_amp, d_fake_amp)
                feats_r1 = feats.detach().requires_grad_(True)
                d_real_fp32 = D(feats_r1)
                r1 = r1_regularizer(d_real_fp32, feats_r1)
                lossD_total = lossD + LAMBDA_R1 * r1

                scalerD.scale(lossD_total).backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                scalerD.step(optD)
                scalerD.update()
                schedD.step()

            # ====== G ======
            optG.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=AMP_ENABLED and device == "cuda"):
                delta = tanh(G(feats)) * delta_scale
                feats_fake = feats + delta
                g_gan = wgan_g_loss(D(feats_fake))
                g_spec = spec_l1(feats_fake, feats)
                lossG = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec

            scalerG.scale(lossG).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            scalerG.step(optG)
            scalerG.update()
            did_stepG = True
            if did_stepG:
                schedG.step()

            ema.update(G)

            if step % LOG_INTERVAL == 0:
                print(f"[ep {ep}] step {step}  "
                      f"D={lossD.item():.3f}  G={lossG.item():.3f}  "
                      f"g_gan={g_gan.item():.3f}  g_spec={g_spec.item():.3f}  r1={r1.item():.3f}")

        # ====== Val ======
        if (ep + 1) % VAL_INTERVAL == 0 and len(val_loader) > 0:
            G.eval()
            total = 0.0; n = 0
            with torch.inference_mode():
                for batch in val_loader:
                    feats = batch["feats"].to(device)
                    feats = torch.nan_to_num(feats, 0.0, 1e6, -1e6)
                    feats_fake = feats + tanh(G(feats)) * delta_scale
                    total += spec_l1(feats_fake, feats).item()
                    n += 1
            print(f"[val] epoch={ep} spec_l1={total/max(1,n):.4f}")

        # checkpoint EMA
        ckpt = {"state_dict": ema.shadow.state_dict(), "c_in": c_in}
        (SAVE_DIR / f"G_ema_ep{ep:03d}.pt").write_bytes(torch.save(ckpt, SAVE_DIR / f"G_ema_ep{ep:03d}.pt") or b"")  # compat py<3.8

if __name__ == "__main__":
    main()
