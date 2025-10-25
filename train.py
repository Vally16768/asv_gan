# train.py
from __future__ import annotations
import os
import argparse
import torch
from torch.utils.data import DataLoader
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # speechbrain așteaptă [B, T]; deja avem [1, T] per eșantion, dar batch=dynamic
        for i in range(wav.size(0)):
            emb = embedder.encode_batch(wav[i:i+1].cpu(), torch.tensor([wav.size(1)]))
            out.append(emb.squeeze(0).squeeze(0))
        return torch.stack(out).to(DEVICE)

def make_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def train(args):
    make_dirs()

    # Dataset + Loader
    ds = BonaFideDataset(args.bona_fide_dir, max_sec=args.max_sec, shuffle=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Modele
    G = UNetMel(n_mels=N_MELS).to(DEVICE)
    D = WDiscriminator(n_mels=N_MELS).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # Vocoder / Detectoare / Speaker
    if not os.path.exists(HIFIGAN_JIT):
        print(f"[vocoder] HiFi-GAN JIT not found at {HIFIGAN_JIT}. Vocoder is only needed when computing evasion/speaker losses or saving audio.")
    vocoder = HiFiGANVocoder(HIFIGAN_JIT, str(DEVICE))
    detectors = load_detectors(DETECTOR_PATHS, str(DEVICE))
    if not detectors:
        print("[detectors] No detectors loaded. Evasion loss will be 0.")
    embedder = get_speaker_embedder()

    logs = []
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            mel = batch["mel"].to(DEVICE)      # [B, n_mels, Tm]
            wave = batch["wave"].to(DEVICE)    # [B, 1, Tw]

            # ----- Train D -----
            for _ in range(args.critic_iters):
                G.eval(); D.train()
                optD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    delta = G(mel)             # [B, n_mels, Tm]
                    mel_fake = mel + delta
                real_scores = D(mel)
                fake_scores = D(mel_fake)
                d_loss = wgan_d_loss(real_scores, fake_scores)
                gp = gradient_penalty(D, mel, mel_fake, DEVICE)
                lossD = d_loss + args.gp_lambda * gp
                lossD.backward()
                optD.step()

            # ----- Train G -----
            G.train(); D.eval()
            optG.zero_grad(set_to_none=True)
            delta = G(mel)
            mel_fake = mel + delta

            fake_scores = D(mel_fake)
            g_gan = wgan_g_loss(fake_scores)           # adversarial
            g_spec = spec_l1(mel_fake, mel)            # recon pe mel

            # Evasion loss (doar dacă avem detectoare și vocoder)
            g_evasion = torch.tensor(0.0, device=DEVICE)
            if detectors and vocoder.is_ready():
                with torch.inference_mode():
                    wav_fake = vocoder(mel_fake)       # [B, 1, Tw]
                scores = run_ensemble_detectors(detectors, wave=wav_fake, mel=mel_fake)
                g_evasion = evasion_loss_ensemble(scores, target=0.0, reduce="mean")

            # Speaker loss (opțional, cere vocoder + speechbrain)
            g_spk = torch.tensor(0.0, device=DEVICE)
            if USE_SPEAKER_LOSS and embedder is not None and vocoder.is_ready():
                with torch.inference_mode():
                    wav_fake = vocoder(mel_fake)
                emb_ref = wav_to_emb(embedder, wave.squeeze(1))     # [B, T]
                emb_fake = wav_to_emb(embedder, wav_fake.squeeze(1))
                if emb_ref is not None and emb_fake is not None:
                    g_spk = speaker_loss(emb_fake, emb_ref)

            lossG = args.lambda_gan * g_gan + args.lambda_spec * g_spec + args.lambda_evasion * g_evasion + LAMBDA_SPK * g_spk
            lossG.backward()
            optG.step()

            step += 1
            pbar.set_postfix({
                "D": f"{lossD.item():.3f}",
                "G": f"{lossG.item():.3f}",
                "ev": f"{g_evasion.item():.3f}" if torch.is_tensor(g_evasion) else "0.000"
            })

            logs.append({
                "step": step, "epoch": epoch,
                "lossD": float(lossD.item()),
                "lossG": float(lossG.item()),
                "loss_spec": float(g_spec.item()),
                "loss_evasion": float(g_evasion.item()) if torch.is_tensor(g_evasion) else 0.0,
                "loss_spk": float(g_spk.item()) if torch.is_tensor(g_spk) else 0.0
            })

            # Save periodic
            if step % args.save_every == 0:
                gpath = os.path.join(CHECKPOINT_DIR, f"G_step{step}.pt")
                dpath = os.path.join(CHECKPOINT_DIR, f"D_step{step}.pt")
                torch.save(G.state_dict(), gpath)
                torch.save(D.state_dict(), dpath)
                # sample wave (doar dacă vocoder e disponibil)
                if vocoder.is_ready():
                    with torch.inference_mode():
                        wav_fake = vocoder(mel_fake[:1])  # [1, 1, Tw]
                    save_wav(os.path.join(CHECKPOINT_DIR, f"sample_gen_{step}.wav"), wav_fake[0])
                    save_wav(os.path.join(CHECKPOINT_DIR, f"sample_clean_{step}.wav"), wave[0])

        # log CSV după fiecare epocă (append-safe)
        import pandas as pd
        import json
        log_path = os.path.join(LOG_DIR, "training_log.csv")
        pd.DataFrame(logs).to_csv(log_path, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bona_fide_dir", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--critic_iters", type=int, default=CRITIC_ITERS)
    ap.add_argument("--gp_lambda", type=float, default=GP_LAMBDA)
    ap.add_argument("--lambda_gan", type=float, default=LAMBDA_GAN)
    ap.add_argument("--lambda_spec", type=float, default=LAMBDA_SPEC)
    ap.add_argument("--lambda_evasion", type=float, default=LAMBDA_EVASION)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_sec", type=float, default=MAX_SEC)
    args = ap.parse_args()
    train(args)
