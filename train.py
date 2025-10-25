# train.py
from __future__ import annotations
import os
import random
from typing import Dict, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
import pandas as pd
from tqdm import tqdm

from dataset import BonaFideDataset
from models import UNetMel, WDiscriminator
from losses import (
    gradient_penalty, wgan_d_loss, wgan_g_loss,
    evasion_loss_ensemble, spec_l1, speaker_loss
)
from utils import HiFiGANVocoder, save_wav, load_detectors, run_ensemble_detectors
from constants import (
    N_MELS, EPOCHS, BATCH_SIZE, LR, CRITIC_ITERS, GP_LAMBDA,
    LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_EVASION, LAMBDA_SPK,
    MAX_SEC, CHECKPOINT_DIR, LOG_DIR, HIFIGAN_JIT, DETECTOR_PATHS,
    USE_SPEAKER_LOSS
)

# ======= Hardcoded data root =======
BONA_FIDE_DIR = Path("database/data").resolve()

# ======= Internal runtime config (no CLI) =======
VAL_RATIO = 0.10        # 10% validare
DL_NUM_WORKERS = 4
PIN_MEMORY = True
AMP_ENABLED = True      # mixed precision ON
ES_PATIENCE = 5         # early stopping patience (epoci)
LR_RED_FACTOR = 0.5     # ReduceLROnPlateau factor
LR_RED_PATIENCE = 2     # ReduceLROnPlateau patience (epoci)
MIN_LR = 1e-6
SAVE_EVERY = 500        # pas salvare periodica (iteratii)
SAVE_BEST_NAME_G = "G_best.pt"
SAVE_BEST_NAME_D = "D_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_speaker_embedder():
    if not USE_SPEAKER_LOSS:
        return None
    try:
        from speechbrain.pretrained import SpeakerRecognition
        recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa"
        )
        recognizer.mods.eval()
        return recognizer
    except Exception as e:
        print("[speaker] Could not load speechbrain ECAPA:", e)
        return None


def wav_to_emb(embedder, wav):
    if embedder is None:
        return None
    with torch.inference_mode():
        out = []
        # wav: [B, 1, T] sau [B, T]; asiguram [B, T]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        for i in range(wav.size(0)):
            emb = embedder.encode_batch(wav[i:i+1].cpu(), torch.tensor([wav.size(1)]))
            out.append(emb.squeeze(0).squeeze(0))
        return torch.stack(out).to(DEVICE)


def make_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


@torch.inference_mode()
def evaluate(
    dl_val: DataLoader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    vocoder: HiFiGANVocoder,
    detectors,
    embedder
) -> Dict[str, float]:
    """Metice pe setul de validare, fara gradient."""
    G.eval()
    D.eval()

    tot_lossD = 0.0
    tot_lossG = 0.0
    tot_spec = 0.0
    tot_evasion = 0.0
    tot_spk = 0.0
    n = 0

    for batch in dl_val:
        mel = batch["mel"].to(DEVICE, non_blocking=True)    # [B, n_mels, Tm]
        wave = batch["wave"].to(DEVICE, non_blocking=True)  # [B, 1, Tw]

        with autocast(enabled=AMP_ENABLED):
            delta = G(mel)
            mel_fake = mel + delta

            real_scores = D(mel)
            fake_scores = D(mel_fake)
            d_loss = wgan_d_loss(real_scores, fake_scores)

            g_gan = wgan_g_loss(fake_scores)
            g_spec = spec_l1(mel_fake, mel)

            g_evasion = mel.new_zeros(())
            if detectors and vocoder.is_ready():
                wav_fake = vocoder(mel_fake)
                scores = run_ensemble_detectors(detectors, wave=wav_fake, mel=mel_fake)
                g_evasion = evasion_loss_ensemble(scores, target=0.0, reduce="mean")

            g_spk = mel.new_zeros(())
            if USE_SPEAKER_LOSS and embedder is not None and vocoder.is_ready():
                wav_fake = vocoder(mel_fake)
                emb_ref = wav_to_emb(embedder, wave)
                emb_fake = wav_to_emb(embedder, wav_fake)
                if emb_ref is not None and emb_fake is not None:
                    g_spk = speaker_loss(emb_fake, emb_ref)

            lossG = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec + LAMBDA_EVASION * g_evasion + LAMBDA_SPK * g_spk

        bsz = mel.size(0)
        n += bsz
        tot_lossD += d_loss.item() * bsz
        tot_lossG += lossG.item() * bsz
        tot_spec += g_spec.item() * bsz
        tot_evasion += (g_evasion.item() if torch.is_tensor(g_evasion) else 0.0) * bsz
        tot_spk += (g_spk.item() if torch.is_tensor(g_spk) else 0.0) * bsz

    return {
        "val_lossD": tot_lossD / max(n, 1),
        "val_lossG": tot_lossG / max(n, 1),
        "val_loss_spec": tot_spec / max(n, 1),
        "val_loss_evasion": tot_evasion / max(n, 1),
        "val_loss_spk": tot_spk / max(n, 1),
    }


def train():
    set_seed(42)
    make_dirs()

    # ===== Path check =====
    if not BONA_FIDE_DIR.exists():
        cwd = Path().resolve()
        raise FileNotFoundError(
            f"'{BONA_FIDE_DIR}' nu exista (cwd={cwd}). "
            f"Creeaza-l sau fa un link catre folderul cu audio bona-fide."
        )

    # ===== Dataset + split =====
    full_ds = BonaFideDataset(str(BONA_FIDE_DIR), max_sec=MAX_SEC, shuffle=True)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, BATCH_SIZE),
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    # ===== Modele / Optimizatoare =====
    G = UNetMel(n_mels=N_MELS).to(DEVICE)
    D = WDiscriminator(n_mels=N_MELS).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

    # ===== Schedulers (ReduceLROnPlateau pe val_lossG) =====
    schG = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optG, mode="min", factor=LR_RED_FACTOR, patience=LR_RED_PATIENCE, min_lr=MIN_LR, verbose=True
    )
    schD = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optD, mode="min", factor=LR_RED_FACTOR, patience=LR_RED_PATIENCE, min_lr=MIN_LR, verbose=True
    )

    # ===== Vocoder / Detectoare / Speaker =====
    if not os.path.exists(HIFIGAN_JIT):
        print(f"[vocoder] HiFi-GAN JIT not found at {HIFIGAN_JIT}. Vocoder is only needed when computing evasion/speaker losses or saving audio.")
    vocoder = HiFiGANVocoder(HIFIGAN_JIT, str(DEVICE))
    detectors = load_detectors(DETECTOR_PATHS, str(DEVICE))
    if not detectors:
        print("[detectors] No detectors loaded. Evasion loss will be 0.")
    embedder = get_speaker_embedder()

    scalerG = GradScaler(enabled=AMP_ENABLED)
    scalerD = GradScaler(enabled=AMP_ENABLED)

    logs = []
    step = 0

    # ===== Early Stopping state =====
    best_val = float("inf")
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        G.train()
        D.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}")

        for batch in pbar:
            mel = batch["mel"].to(DEVICE, non_blocking=True)      # [B, n_mels, Tm]
            wave = batch["wave"].to(DEVICE, non_blocking=True)    # [B, 1, Tw]

            # ===== Train D =====
            for _ in range(CRITIC_ITERS):
                G.eval(); D.train()
                optD.zero_grad(set_to_none=True)
                with autocast(enabled=AMP_ENABLED):
                    with torch.no_grad():
                        delta = G(mel)             # [B, n_mels, Tm]
                        mel_fake = mel + delta
                    real_scores = D(mel)
                    fake_scores = D(mel_fake)
                    d_loss = wgan_d_loss(real_scores, fake_scores)
                    gp = gradient_penalty(D, mel, mel_fake, DEVICE)
                    lossD = d_loss + GP_LAMBDA * gp
                scalerD.scale(lossD).backward()
                scalerD.step(optD)
                scalerD.update()

            # ===== Train G =====
            G.train(); D.eval()
            optG.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                delta = G(mel)
                mel_fake = mel + delta

                fake_scores = D(mel_fake)
                g_gan = wgan_g_loss(fake_scores)           # adversarial
                g_spec = spec_l1(mel_fake, mel)            # recon pe mel

                g_evasion = mel.new_zeros(())
                if detectors and vocoder.is_ready():
                    with torch.inference_mode():
                        wav_fake = vocoder(mel_fake)       # [B, 1, Tw]
                    scores = run_ensemble_detectors(detectors, wave=wav_fake, mel=mel_fake)
                    g_evasion = evasion_loss_ensemble(scores, target=0.0, reduce="mean")

                g_spk = mel.new_zeros(())
                if USE_SPEAKER_LOSS and embedder is not None and vocoder.is_ready():
                    with torch.inference_mode():
                        wav_fake = vocoder(mel_fake)
                    emb_ref = wav_to_emb(embedder, wave)     # [B, T]
                    emb_fake = wav_to_emb(embedder, wav_fake)
                    if emb_ref is not None and emb_fake is not None:
                        g_spk = speaker_loss(emb_fake, emb_ref)

                lossG = LAMBDA_GAN * g_gan + LAMBDA_SPEC * g_spec + LAMBDA_EVASION * g_evasion + LAMBDA_SPK * g_spk

            scalerG.scale(lossG).backward()
            scalerG.step(optG)
            scalerG.update()

            step += 1
            pbar.set_postfix({
                "D": f"{lossD.item():.3f}",
                "G": f"{lossG.item():.3f}",
                "ev": f"{(g_evasion.item() if torch.is_tensor(g_evasion) else 0.0):.3f}"
            })

            logs.append({
                "step": step, "epoch": epoch,
                "lossD": float(lossD.item()),
                "lossG": float(lossG.item()),
                "loss_spec": float(g_spec.item()),
                "loss_evasion": float(g_evasion.item()) if torch.is_tensor(g_evasion) else 0.0,
                "loss_spk": float(g_spk.item()) if torch.is_tensor(g_spk) else 0.0,
                "lr_G": float(optG.param_groups[0]["lr"]),
                "lr_D": float(optD.param_groups[0]["lr"]),
                "phase": "train"
            })

            # Save periodic
            if step % SAVE_EVERY == 0:
                gpath = os.path.join(CHECKPOINT_DIR, f"G_step{step}.pt")
                dpath = os.path.join(CHECKPOINT_DIR, f"D_step{step}.pt")
                torch.save(G.state_dict(), gpath)
                torch.save(D.state_dict(), dpath)
                if vocoder.is_ready():
                    with torch.inference_mode():
                        wav_fake = vocoder(mel_fake[:1])  # [1, 1, Tw]
                    save_wav(os.path.join(CHECKPOINT_DIR, f"sample_gen_{step}.wav"), wav_fake[0])
                    save_wav(os.path.join(CHECKPOINT_DIR, f"sample_clean_{step}.wav"), wave[0])

        # ===== Validare =====
        val_metrics = evaluate(dl_val, G, D, vocoder, detectors, embedder)
        logs.append({
            "step": step, "epoch": epoch,
            **val_metrics,
            "lr_G": float(optG.param_groups[0]["lr"]),
            "lr_D": float(optD.param_groups[0]["lr"]),
            "phase": "val"
        })

        # CSV (rescriere simpla)
        log_path = os.path.join(LOG_DIR, "training_log.csv")
        pd.DataFrame(logs).to_csv(log_path, index=False)

        # ===== LR scheduling pe baza val_lossG =====
        schG.step(val_metrics["val_lossG"])
        schD.step(val_metrics["val_lossG"])

        # ===== Early Stopping + best checkpoint =====
        current = val_metrics["val_lossG"]
        improved = current < best_val - 1e-6
        if improved:
            best_val = current
            no_improve_epochs = 0
            torch.save(G.state_dict(), os.path.join(CHECKPOINT_DIR, SAVE_BEST_NAME_G))
            torch.save(D.state_dict(), os.path.join(CHECKPOINT_DIR, SAVE_BEST_NAME_D))
        else:
            no_improve_epochs += 1

        print(f"[val] epoch={epoch} val_lossG={current:.4f} best={best_val:.4f} no_improve={no_improve_epochs}")
        if no_improve_epochs >= ES_PATIENCE:
            print(f"[early-stopping] Stop dupa {ES_PATIENCE} epoci fara imbunatatire.")
            break


if __name__ == "__main__":
    # No CLI args — run with hardcoded path & constants
    train()
