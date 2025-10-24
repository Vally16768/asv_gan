# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import BonaFideDataset, mel_transform
from models import UNetMel, WDiscriminator
from losses import *
from utils import mel_to_wave, save_wav, load_detector
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    ds = BonaFideDataset(args.bona_fide_dir, max_sec=args.max_sec)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    G = UNetMel(n_mels=80).to(DEVICE)
    D = WDiscriminator(n_mels=80).to(DEVICE)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    detector = None
    if args.detector_path:
        detector = load_detector(args.detector_path, DEVICE)
    # optional speaker embedding model (not included here)

    logs = []
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl)
        for batch in pbar:
            mel = batch["mel"].to(DEVICE)  # [B, n_mels, T]
            wave = batch["wave"].to(DEVICE)  # [B, L]
            B = mel.size(0)

            # --------------------
            # Train discriminator (WGAN)
            # --------------------
            for _ in range(args.critic_iters):
                G.eval()
                D.train()
                optD.zero_grad()
                with torch.no_grad():
                    delta = G(mel).detach()
                    gen_mel = mel + delta
                real_scores = D(mel)
                fake_scores = D(gen_mel)
                d_loss = wgan_d_loss(real_scores, fake_scores)
                gp = gradient_penalty(D, mel.unsqueeze(1), gen_mel.unsqueeze(1), DEVICE)
                lossD = d_loss + args.gp_lambda * gp
                lossD.backward()
                optD.step()

            # --------------------
            # Train generator
            # --------------------
            G.train()
            D.eval()
            optG.zero_grad()
            delta = G(mel)
            gen_mel = mel + delta
            # GAN loss
            fake_scores = D(gen_mel)
            loss_g_gan = wgan_g_loss(fake_scores)
            # spectral loss
            loss_spec = spec_loss(gen_mel, mel)
            # evasion loss via detector
            loss_evasion = torch.tensor(0.0, device=DEVICE)
            if detector is not None:
                # try waveform path first: reconstruct waveform from mel (slow). For speed, optionally detector can accept mel.
                # convert gen_mel to waveform
                waves = mel_to_wave(gen_mel.detach().cpu())  # [B, L]
                waves = waves.to(DEVICE)
                try:
                    det_scores = detector(waves)  # expect [B] or [B,1]
                except Exception:
                    # try detector accepts mel input
                    try:
                        det_scores = detector(gen_mel)
                    except Exception:
                        raise RuntimeError("Detector does not accept wave or mel tensors directly. Provide wrapper.")
                if det_scores.dim() > 1:
                    det_scores = det_scores.squeeze(-1)
                # we want detector to predict bona fide (0) -> minimize MSE to 0
                loss_evasion = evasion_loss_from_detector(det_scores, target_label=0.0)

            # total loss
            lossG = args.lambda_gan * loss_g_gan + args.lambda_spec * loss_spec + args.lambda_evasion * loss_evasion
            lossG.backward()
            optG.step()

            # logging
            step += 1
            pbar.set_description(f"Epoch {epoch} Dloss {lossD.item():.4f} Gloss {lossG.item():.4f} Ev {loss_evasion.item() if hasattr(loss_evasion, 'item') else loss_evasion:.4f}")
            logs.append({
                "step": step,
                "epoch": epoch,
                "lossD": lossD.item(),
                "lossG": lossG.item(),
                "loss_spec": loss_spec.item(),
                "loss_evasion": loss_evasion.item() if hasattr(loss_evasion, "item") else 0.0
            })

            # checkpoint periodic save and example outputs
            if step % args.save_every == 0:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(G.state_dict(), os.path.join(args.ckpt_dir, f"G_step{step}.pt"))
                torch.save(D.state_dict(), os.path.join(args.ckpt_dir, f"D_step{step}.pt"))
                # save first item waveform for quick listening
                gen_wave = mel_to_wave(gen_mel[0:1].detach().cpu())[0]
                save_wav(os.path.join(args.ckpt_dir, f"sample_gen_{step}.wav"), gen_wave.numpy())
                save_wav(os.path.join(args.ckpt_dir, f"sample_clean_{step}.wav"), batch["wave"][0].cpu().numpy())

        # save epoch logs
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(args.ckpt_dir, "training_log.csv"), index=False)

    print("Training finished. Checkpoints in", args.ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bona_fide_dir", required=True)
    parser.add_argument("--detector_path", default=None, help="path to your pretrained detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--critic_iters", type=int, default=5)
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_spec", type=float, default=10.0)
    parser.add_argument("--lambda_evasion", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--max_sec", type=float, default=6.0)
    args = parser.parse_args()
    train(args)
