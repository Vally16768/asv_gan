# train.py
import os, argparse, torch, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BonaFideDataset
from models import UNetMel, WDiscriminator
from losses import gradient_penalty, wgan_d_loss, wgan_g_loss, evasion_loss_ensemble, spec_l1, speaker_loss
from utils import HiFiGANVocoder, save_wav, load_detectors, run_ensemble_detectors
from constants import (SR, N_MELS, EPOCHS, BATCH_SIZE, LR, CRITIC_ITERS, GP_LAMBDA,
                       LAMBDA_GAN, LAMBDA_SPEC, LAMBDA_EVASION, LAMBDA_SPK,
                       CHECKPOINT_DIR, LOG_DIR, MAX_SEC, HIFIGAN_JIT, DETECTOR_PATHS,
                       USE_SPEAKER_LOSS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- optional speaker embedder (ECAPA via speechbrain) ----
def get_speaker_embedder():
    if not USE_SPEAKER_LOSS:
        return None
    try:
        from speechbrain.pretrained import SpeakerRecognition
        recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir="pretrained_ecapa")
        recognizer.mods.eval()
        return recognizer
    except Exception as e:
        print("[speaker] Could not load speechbrain ECAPA:", e)
        return None

def wav_to_emb(embedder, wav):
    # wav: [B, L] @16k; returns [B, D]
    if embedder is None: return None
    with torch.inference_mode():
        # speechbrain forward handles batch internally; pass list of tensors
        out = []
        for i in range(wav.size(0)):
            emb = embedder.encode_batch(wav[i:i+1].cpu(), torch.tensor([wav.size(1)]))
            out.append(emb.squeeze(0).squeeze(0))
        return torch.stack(out).to(DEVICE)

def make_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def train(args):
    make_dirs()
    ds = BonaFideDataset(args.bona_fide_dir, max_sec=args.max_sec)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    G = UNetMel(n_mels=N_MELS).to(DEVICE)
    D = WDiscriminator(n_mels=N_MELS).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # HiFi-GAN (TorchScript) pentru reconstrucție rapidă wave
    vocoder = HiFiGANVocoder(HIFIGAN_JIT, DEVICE)

    # Ensemble de detectoare
    detectors = load_detectors(DETECTOR_PATHS, DEVICE)

    # Speaker embedder (opțional)
    embedder = get_speaker_embedder()

    logs = []
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            mel = batch["mel"].to(DEVICE)       # [B, n_mels, T]
            wave = batch["wave"].to(DEVICE)     # [B, L]

            # ----- Train D -----
            for _ in range(args.critic_iters):
                G.eval(); D.train()
                optD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    delta = G(mel)
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

            # GAN loss
            fake_scores = D(mel_fake)
            g_gan = wgan_g_loss(fake_scores)

            # spectral
            g_spec = spec_l1(mel_fake, mel)

            # evasion via ensemble
            scores = []
            if detectors:
                with torch.inference_mode():
                    wav_fake = vocoder(mel_fake)   # [B, L] rapid
                scores = run_ensemble_detectors(detectors, wave=wav_fake, mel=mel_fake)
            g_evasion = evasion_loss_ensemble(scores, target=0.0, reduce="mean") if detectors else torch.tensor(0.0, device=DEVICE)

            # speaker (opțional)
            g_spk = torch.tensor(0.0, device=DEVICE)
            if USE_SPEAKER_LOSS and embedder is not None:
                with torch.inference_mode():
                    wav_fake = vocoder(mel_fake)
                emb_ref = wav_to_emb(embedder, wave)
                emb_fake = wav_to_emb(embedder, wav_fake)
                if emb_ref is not None and emb_fake is not None:
                    g_spk = speaker_loss(emb_fake, emb_ref)

            lossG = args.lambda_gan * g_gan + args.lambda_spec * g_spec + args.lambda_evasion * g_evasion + LAMBDA_SPK * g_spk
            lossG.backward()
            optG.step()

            step += 1
            pbar.set_postfix({
                "D": f"{lossD.item():.3f}",
                "G": f"{lossG.item():.3f}",
                "ev": f"{g_evasion.item():.3f}"
            })

            logs.append({
                "step": step, "epoch": epoch,
                "lossD": lossD.item(), "lossG": lossG.item(),
                "loss_spec": g_spec.item(),
                "loss_evasion": g_evasion.item() if torch.is_tensor(g_evasion) else 0.0,
                "loss_spk": g_spk.item() if torch.is_tensor(g_spk) else 0.0
            })

            if step % args.save_every == 0:
                gpath = os.path.join(CHECKPOINT_DIR, f"G_step{step}.pt")
                dpath = os.path.join(CHECKPOINT_DIR, f"D_step{step}.pt")
                torch.save(G.state_dict(), gpath)
                torch.save(D.state_dict(), dpath)
                # salvează exemple
                with torch.inference_mode():
                    wav_fake = vocoder(mel_fake[:1])
                save_wav(os.path.join(CHECKPOINT_DIR, f"sample_gen_{step}.wav"), wav_fake[0])
                save_wav(os.path.join(CHECKPOINT_DIR, f"sample_clean_{step}.wav"), wave[0])

        pd.DataFrame(logs).to_csv(os.path.join(LOG_DIR, "training_log.csv"), index=False)

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
    ap.add_argument("--save_every", type=int, default=SAVE_EVERY)
    ap.add_argument("--max_sec", type=float, default=MAX_SEC)
    args = ap.parse_args()
    train(args)
