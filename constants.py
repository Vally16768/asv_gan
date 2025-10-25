# constants.py
from pathlib import Path

# Audio constants
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_SEC = 8.0  # maxim lungime audio (sec) pentru pad/truncate

# Training defaults
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-3

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "database" / "data"
BONA_ONLY_DIR = PROJECT_ROOT / "database" / "bonafide_only"
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"

# Device (optional override)
DEVICE = "cuda"  # or "cpu" (auto-detected in scripts)
