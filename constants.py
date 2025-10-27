# constants.py — tuned for stronger WGAN training (defensive / research)
from pathlib import Path

# ----------------- Audio / Features -----------------
SR = 16000
# lungimea maximă a unui exemplu în antrenare (sec)
MAX_AUDIO_SECONDS = 4
SEGMENT_SAMPLES   = MAX_AUDIO_SECONDS * SR

N_MELS = 160
N_FFT = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k
FMIN = 20
FMAX = 7600
POWER = 1.0           # magnitude spectrogram power for mel

FEATS_MEAN = 0.0
FEATS_STD  = 1.0

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"
SAVE_DIR = ROOT / "runs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Training -----------------
AMP_ENABLED = True            # mixed precision for speed/stability
BATCH_SIZE = 6
EPOCHS = 1000                 # high cap; early stopping will typically stop earlier

# TTUR (Two Time-scale) — D slightly faster than G
LR_G = 1e-4
LR_D = 2e-4
BETA1 = 0.0
BETA2 = 0.99
WEIGHT_DECAY = 0.0

# More critic updates initially -> stronger critic signal
CRITIC_ITERS = 3
GRAD_CLIP = 5.0

# Loss weights
LAMBDA_GAN  = 1.0
LAMBDA_SPEC = 3.0
LAMBDA_FM   = 1.0
# stronger R1 regularization (applied lazily)
LAMBDA_R1   = 10.0
R1_EVERY    = 16

# Schedules for instance noise & dithering (applied to inputs of D / spectrograms)
DELTA_INIT  = 0.02
DELTA_MIN   = 0.002
DELTA_DECAY = 0.9999  # slower decay

INST_NOISE_INIT  = 0.02
INST_NOISE_MIN   = 0.0
INST_NOISE_DECAY = 0.9999  # slower decay to stabilize longer

# Evasion warm-up
EVASION_WARMUP_STEPS = 5000
EVASION_RAMP_STEPS   = 15000
LAMBDA_EVASION_MAX   = 1.0

# Logging / saving
LOG_INTERVAL = 100
VAL_INTERVAL = 999999999  # not using validation by default
SAVE_AUDIO_EVERY_EPOCH = True
SAMPLES_DIR = SAVE_DIR / "samples"
CKPT_DIR    = SAVE_DIR / "checkpoints"
LOG_CSV     = SAVE_DIR / "train_log.csv"
BEST_BY     = "p_bona_mean"  # primary metric

# Misc
SEED = 1337
NUM_WORKERS = 6
PIN_MEMORY = True

# EMA
USE_EMA = True
EMA_DECAY = 0.999

# Surrogate ASV (optional but recommended)
USE_SURROGATE = True
SURROGATE_LR = 2e-4
SURROGATE_BETA1 = 0.9
SURROGATE_BETA2 = 0.999
SURROGATE_W = 1.0  # weight inside evasion term
SURROGATE_UPDATE_EVERY = 5  # how often to update surrogate (global steps)

# Early stop target (ASV bona_fide probability)
TARGET_P_BONA = 0.80
TARGET_WINDOW = 20        # rolling window (number of logged entries) for smoothing
MIN_STEPS_TO_CHECK = 2000 # don't consider early stopping earlier than this many steps

# Safety caps
MAX_TRAIN_STEPS = 5_000_000

# ----------------- ASVspoof strict adapter (infer_ahkmno source of truth) -----------------
ASV_COMBO = "AHKMNO"
ASV_MODEL_PATH = ROOT / "ASVmodel" / "best_model.keras"
ASV_SCALER_PATH = ROOT / "ASVmodel" / "scaler.pkl"
ASV_TMP_DIR = SAVE_DIR / "_tmp_asv"
ASV_TMP_DIR.mkdir(parents=True, exist_ok=True)
