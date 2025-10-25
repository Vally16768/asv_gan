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

# WGAN / loss hyperparams (defaults - can be overridden via CLI)
CRITIC_ITERS = 5
GP_LAMBDA = 10.0
LAMBDA_GAN = 1.0
LAMBDA_SPEC = 10.0
LAMBDA_EVASION = 1.0
LAMBDA_SPK = 0.1

# Save / logging
SAVE_EVERY = 500
CHECKPOINT_DIR = Path(__file__).resolve().parent / "runs" / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "runs" / "logs"
EVAL_OUT = Path(__file__).resolve().parent / "runs" / "eval"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "database" / "data"
BONA_ONLY_DIR = PROJECT_ROOT / "database" / "bonafide_only"

# HiFi-GAN / detector models (set sensible defaults; override via env/CLI)
HIFIGAN_JIT = PROJECT_ROOT / "pretrained" / "hifigan.jit"  # change to your jit path
DETECTOR_PATHS = [
    # paths to detectors used by ensemble. Example:
    # str(PROJECT_ROOT / "ASVmodel" / "best_model.keras")
]

# Device (optional override)
DEVICE = "cuda"  # or "cpu" (auto-detected in scripts)

# Speaker loss toggle
USE_SPEAKER_LOSS = False
