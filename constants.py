# constants.py
from __future__ import annotations
from pathlib import Path

# ----------------- Audio / Features -----------------
SR = 16000
TARGET_SR = SR

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k

# Normalizare features (poți ajusta după o trecere pe date)
FEATS_MEAN = 0.0
FEATS_STD  = 1.0

# ----------------- Căi -----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"

# ----------------- Antrenare -----------------
AMP_ENABLED = True
BATCH_SIZE = 32
EPOCHS = 3
CRITIC_ITERS = 5

# TTUR
LR_G = 2e-4
LR_D = 4e-4
BETA1 = 0.0
BETA2 = 0.99

# Ponderi pierderi (mai conservator pentru stabilitate inițială)
LAMBDA_GAN = 1.0
LAMBDA_SPEC = 1.0        # redus de la 10.0
LAMBDA_R1 = 5.0          # redus de la 10.0

# ----------------- Logging -----------------
LOG_INTERVAL = 50
VAL_INTERVAL = 1
SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
