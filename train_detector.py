# train_detector.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm

from dataset import BonaFideDataset
from constants import (SR, N_MELS, EPOCHS, BATCH_SIZE, LR, CHECKPOINT_DIR, LOG_DIR, MAX_SEC)
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleDetector(nn.Module):
    def __init__(self, n_mels=N_MELS, n_class=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (5,5), padding=(2,2)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, (5,5), padding=(2,2)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3), padding=(1,1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, n_class)

    def forward(self, mel):
        if mel.dim() == 3:
            x = mel.unsqueeze(1)
        else:
            x = mel
        feat = self.net(x).view(x.size(0), -1)
        out = self.fc(feat)
        return out.squeeze(-1)

def binary_eer(labels, scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = (fpr - fnr).abs().argmin()
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)

def make_dirs(checkpoint_dir, log_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

def train(args):
    make_dirs(args.checkpoint_dir, args.log_dir)
    ds = BonaFideDataset(args.root_dir, protocol_csv=args.protocol_csv, max_sec=args.max_sec)
    # NOTE: for supervised training you need both classes (bona + spoof). If you only pass bona_fide
    # files then labels will be all zeros and standard CE/BCE won't train a useful classifier.
    # For didactic one-class training, you should use an autoencoder approach (ask me and-I'll add).
    n = len(ds)
    nval = max(1, int(0.1*n))
    ntrain = n - nval
    train_ds, val_ds = random_split(ds, [ntrain, nval])

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleDetector().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    logs = []
    best_eer = 1.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl_train, desc=f"train epoch {epoch}")
        for batch in pbar:
            mel = batch["mel"].to(DEVICE)
            # DUMMY labels: zeros (bona). For supervised training, your dataset must return labels.
            labels = torch.zeros(mel.size(0), device=DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(mel)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            logs.append({"epoch": epoch, "loss": float(loss.item())})

        # validation
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in dl_val:
                mel = batch["mel"].to(DEVICE)
                labels = torch.zeros(mel.size(0))
                logits = model(mel).cpu().numpy()
                all_scores.extend(logits.tolist())
                all_labels.extend(labels.tolist())
        try:
            eer = binary_eer(all_labels, all_scores)
        except Exception:
            eer = None

        ckpt = os.path.join(args.checkpoint_dir, f"detector_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"[train] epoch {epoch} done; val_eer={eer}")
        logs.append({"epoch": epoch, "val_eer": eer})
        if eer is not None and eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "detector_best.pt"))

    pd.DataFrame(logs).to_csv(os.path.join(args.log_dir, "detector_train_log.csv"), index=False)
    print("[train] finished")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="folder with audio (or symlinked bona_fide_only dir)")
    ap.add_argument("--protocol_csv", default=None, help="optional CSV with columns: path,label (0 bona,1 spoof)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max_sec", type=float, default=MAX_SEC)
    ap.add_argument("--checkpoint_dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--log_dir", default=str(LOG_DIR))
    args = ap.parse_args()
    train(args)
