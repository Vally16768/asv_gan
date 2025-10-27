
from pathlib import Path

# ----------------- Audio / Features -----------------
SR = 16000
# lungimea maximă a unui exemplu în antrenare
MAX_AUDIO_SECONDS = 3
SEGMENT_SAMPLES   = MAX_AUDIO_SECONDS * SR

N_MELS = 128
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
BATCH_SIZE = 8
EPOCHS = 200

# TTUR (stabil)
LR_G = 1.2e-4
LR_D = 1.5e-4
BETA1 = 0.0
BETA2 = 0.99
WEIGHT_DECAY = 0.0

CRITIC_ITERS = 1             # keep D in check with strong MS architecture
GRAD_CLIP = 5.0              # clip G grads for stability (None to disable)

# Loss weights
LAMBDA_GAN  = 1.0
LAMBDA_SPEC = 2.0            # MR-STFT + log-mel has stronger pull
LAMBDA_FM   = 5.0e-1         # feature matching from D
LAMBDA_R1   = 0.25           # R1 penalty (on real) for WGAN

# Schedules for instance noise & dithering (applied to inputs of D / spectrograms)
DELTA_INIT  = 0.02
DELTA_MIN   = 0.002
DELTA_DECAY = 0.995

INST_NOISE_INIT  = 0.02
INST_NOISE_MIN   = 0.0
INST_NOISE_DECAY = 0.995

# Evasion warm-up
EVASION_WARMUP_STEPS = 3000
EVASION_RAMP_STEPS   = 6000
LAMBDA_EVASION_MAX   = 0.7

# Logging / saving
LOG_INTERVAL = 100
VAL_INTERVAL = 999999999  # set very large if you do not use validation for now
SAVE_AUDIO_EVERY_EPOCH = True
SAMPLES_DIR = SAVE_DIR / "samples"
CKPT_DIR    = SAVE_DIR / "checkpoints"
LOG_CSV     = SAVE_DIR / "train_log.csv"
BEST_BY     = "p_bona_mean"  # or "lossG_val" if using validation

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
SURROGATE_W = 1.0  # loss weight inside evasion term
