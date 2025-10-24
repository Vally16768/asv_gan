# eval.py
import argparse
import torch
from dataset import BonaFideDataset
from models import UNetMel
from utils import mel_to_wave, load_detector, save_wav
from sklearn.metrics import roc_curve
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_eer(labels, scores):
    # labels: 0 bona-fide, 1 spoof. But here we have only bona-fide; for attack eval we treat detector score high=spoof
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # find threshold where fpr ~= fnr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def eval(args):
    ds = BonaFideDataset(args.bona_fide_dir, max_sec=args.max_sec)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    G = UNetMel(n_mels=80).to(DEVICE)
    G.load_state_dict(torch.load(args.generator_ckpt, map_location=DEVICE))
    G.eval()

    detector = load_detector(args.detector_path, DEVICE)

    detector_scores_clean = []
    detector_scores_adv = []
    for batch in tqdm(dl):
        mel = batch["mel"].to(DEVICE)
        wave = batch["wave"].to(DEVICE)
        with torch.no_grad():
            delta = G(mel)
            gen_mel = mel + delta
            # reconstruct waves
            gen_wave = mel_to_wave(gen_mel.cpu())[0].to(DEVICE)
            # detector may expect wave or mel
            try:
                score_clean = detector(wave.unsqueeze(0))
                score_adv = detector(gen_wave.unsqueeze(0))
            except Exception:
                score_clean = detector(mel)
                score_adv = detector(gen_mel)
            # ensure scalars
            if score_clean.dim() > 0: score_clean = score_clean.squeeze().cpu().item()
            else: score_clean = float(score_clean)
            if score_adv.dim() > 0: score_adv = score_adv.squeeze().cpu().item()
            else: score_adv = float(score_adv)
            detector_scores_clean.append(score_clean)
            detector_scores_adv.append(score_adv)

    # For EER calculation we need labels for positive(=spoof) and negative(=bona). Here we test how detector scores change:
    # Simplest view: compute how many adv scores go below a detection threshold (like median/operating point).
    # For demonstration compute mean scores and report relative drop.
    import numpy as np
    print("clean mean score:", np.mean(detector_scores_clean))
    print("adv mean score:", np.mean(detector_scores_adv))
    # save per-utt csv
    import pandas as pd
    df = pd.DataFrame({
        "path": [b["path"][0] for b in dl.dataset],  # careful: dataset order
        "score_clean": detector_scores_clean,
        "score_adv": detector_scores_adv
    })
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "detector_scores.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bona_fide_dir", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--detector_path", required=True)
    parser.add_argument("--out_dir", default="eval_out")
    parser.add_argument("--max_sec", type=float, default=6.0)
    args = parser.parse_args()
    eval(args)
