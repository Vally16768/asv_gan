import os, argparse, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve

from dataset import BonaFideDataset
from models import UNetMel
from utils import HiFiGANVocoder, load_detectors, run_ensemble_detectors, save_wav
from constants import (CHECKPOINT_DIR, EVAL_OUT, HIFIGAN_JIT, DETECTOR_PATHS, SR, MAX_SEC, N_MELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def eval_model(args):
    os.makedirs(EVAL_OUT, exist_ok=True)
    ds = BonaFideDataset(args.bona_fide_dir, max_sec=args.max_sec)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    G = UNetMel(n_mels=N_MELS).to(DEVICE)
    G.load_state_dict(torch.load(args.generator_ckpt, map_location=DEVICE))
    G.eval()

    vocoder = HiFiGANVocoder(HIFIGAN_JIT, str(DEVICE))
    detectors = load_detectors(DETECTOR_PATHS, str(DEVICE))

    rows = []
    for batch in tqdm(dl, desc="eval"):
        mel = batch["mel"].to(DEVICE)
        wave = batch["wave"].to(DEVICE)
        path = batch["path"][0]

        with torch.inference_mode():
            delta = G(mel)
            mel_fake = mel + delta
            wav_fake = vocoder(mel_fake)

        scores = run_ensemble_detectors(detectors, wave=wav_fake, mel=mel_fake)
        s = torch.stack([x.squeeze() for x in scores], dim=0).mean().item() if scores else float("nan")

        rows.append({"path": path, "score_adv": s})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(EVAL_OUT, "detector_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[eval] Wrote {out_csv}")

    if args.protocol_csv and os.path.exists(args.protocol_csv):
        proto = pd.read_csv(args.protocol_csv)
        merged = df.merge(proto, on="path", how="inner")
        labels = merged["label"].to_numpy().astype(int)
        scores = merged["score_adv"].to_numpy().astype(float)

        valid = np.isfinite(scores)
        s = scores[valid]
        if args.score_higher_is_bonafide:
            s = -s  # inversăm sensul astfel încât 'mai mare' să însemne 'spoof'
        eer = compute_eer(labels[valid], s)
        print(f"[eval] EER={eer:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bona_fide_dir", required=True)
    ap.add_argument("--generator_ckpt", required=True)
    ap.add_argument("--protocol_csv", default=None, help="CSV: path,label(0 bona,1 spoof)")
    ap.add_argument("--score_higher_is_bonafide", action="store_true",
                    help="Setează dacă scorurile detectorului sunt mai mari pentru bona-fide (le inversăm pentru EER).")
    ap.add_argument("--max_sec", type=float, default=MAX_SEC)
    args = ap.parse_args()
    eval_model(args)
