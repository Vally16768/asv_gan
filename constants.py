#constants.py
from __future__ import annotations
from pathlib import Path

# ----------------- Audio / Features -----------------
SR = 16000
TARGET_SR = SR

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k

# Normalizare features (putem rafina ulterior cu stats din set)
FEATS_MEAN = 0.0
FEATS_STD  = 1.0

# ----------------- Căi -----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"

# ----------------- Antrenare -----------------
AMP_ENABLED = False            # <— dezactivat până stabilizăm magnitudinile
BATCH_SIZE = 32
EPOCHS = 3
CRITIC_ITERS = 5

# TTUR (mai conservator)
LR_G = 1e-4
LR_D = 2e-4
BETA1 = 0.0
BETA2 = 0.99

# Ponderi pierderi
LAMBDA_GAN = 1.0
LAMBDA_SPEC = 1.0
LAMBDA_R1 = 1.0              # <— mai mic pentru stabilitate inițială

# ----------------- Logging -----------------
LOG_INTERVAL = 50
VAL_INTERVAL = 1
SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- ASVspoof (black-box evasion) -----------------
ASV_MODEL_DIR   = ROOT / "ASVmodel"          # conține best_model.keras
ASV_SCALER      = ASV_MODEL_DIR / "scaler.pkl"
ASV_COMBO       = "AHKMNO"                   # codul tău default
ASV_SR          = SR

EVASION_LAMBDA  = 1.0     # forță penalizare
EVASION_EVERY   = 5       # o dată la N pași
TARGET_LABEL    = "bona_fide"  # dacă există labels.txt (altfel index=0)
