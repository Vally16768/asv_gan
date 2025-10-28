# constants.py — config stabil pentru DDP + ASV
from pathlib import Path

# ----------------- Audio / Features -----------------
SR = 16000
MAX_AUDIO_SECONDS = 2
SEGMENT_SAMPLES   = MAX_AUDIO_SECONDS * SR

N_MELS = 160
N_FFT = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k
FMIN = 20
FMAX = 7600
POWER = 1.0

FEATS_MEAN = 0.0
FEATS_STD  = 1.0

# ----------------- Paths -----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"

SAVE_DIR = ROOT / "runs"
SAMPLES_DIR = SAVE_DIR / "samples"
CKPT_DIR    = SAVE_DIR / "checkpoints"
LOG_CSV     = SAVE_DIR / "train_log.csv"
for p in (SAVE_DIR, SAMPLES_DIR, CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ----------------- Training -----------------
AMP_ENABLED = True
BATCH_SIZE = 4
EPOCHS = 1000

# TTUR
LR_G = 1e-4
LR_D = 2e-4
BETA1 = 0.0
BETA2 = 0.99
WEIGHT_DECAY = 0.0

CRITIC_ITERS = 3
GRAD_CLIP = 5.0

# Loss weights
LAMBDA_GAN  = 1.0
LAMBDA_SPEC = 3.0
LAMBDA_FM   = 1.0
LAMBDA_R1   = 10.0
R1_EVERY    = 16

# Schedules (stabilize)
DELTA_INIT  = 0.02
DELTA_MIN   = 0.002
DELTA_DECAY = 0.9999

INST_NOISE_INIT  = 0.02
INST_NOISE_MIN   = 0.0
INST_NOISE_DECAY = 0.9999

# Evasion schedule
EVASION_WARMUP_STEPS = 5000
EVASION_RAMP_STEPS   = 15000
LAMBDA_EVASION_MAX   = 1.0

# Logging / saving
LOG_INTERVAL = 100
SAVE_AUDIO_EVERY_EPOCH = True
BEST_BY = "p_bona_mean"

# Misc
SEED = 1337
NUM_WORKERS = 6
PIN_MEMORY = True

# EMA
USE_EMA = True
EMA_DECAY = 0.999

# Surrogate ASV
USE_SURROGATE = True
SURROGATE_LR = 2e-4
SURROGATE_BETA1 = 0.9
SURROGATE_BETA2 = 0.999
SURROGATE_W = 1.0               # weight în loss-ul de evasion
SURROGATE_UPDATE_EVERY = 5      # pași între update-urile surrogate (pe rank-0)

# Early stop (evaluat pe rank-0; aplicat la final de epocă)
TARGET_P_BONA = 0.80
TARGET_WINDOW = 20
MIN_STEPS_TO_CHECK = 2000
MAX_TRAIN_STEPS = 5_000_000

# Keras logging (opțional, mai lent; dacă False, folosește surrogate pt. metrică)
LOG_WITH_KERAS = False

# ----------------- ASVspoof strict adapter -----------------
ASV_COMBO = "AHKMNO"
ASV_MODEL_PATH = ROOT / "ASVmodel" / "best_model.keras"  # sau .h5; auto-detect în loader
ASV_SCALER_PATH = ROOT / "ASVmodel" / "scaler.pkl"
ASV_TMP_DIR = SAVE_DIR / "_tmp_asv"
ASV_TMP_DIR.mkdir(parents=True, exist_ok=True)
