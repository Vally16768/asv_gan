# constants.py — „long-run, natural + ASV evasion” profile
from pathlib import Path

# ----------------- Audio & featuri -----------------
SR         = 16000
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k
FEATS_MEAN = 0.0
FEATS_STD  = 1.0

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"

# ----------------- Antrenare (long-run) -----------------
AMP_ENABLED  = True          # activăm AMP pentru viteză pe run lung
BATCH_SIZE   = 24
EPOCHS       = 200           # long-run; poți opri prin early stopping
CRITIC_ITERS = 3             # un pic mai mare => D mai solid, dar fără să domine

# TTUR (stabil & ușor pro-G)
LR_G  = 1.0e-4
LR_D  = 1.5e-4
BETA1 = 0.0
BETA2 = 0.99

# ----------------- Ponderi pierderi -----------------
LAMBDA_GAN  = 1.0
LAMBDA_SPEC = 20.0           # calitate/naturalness; înainte era 2.0
LAMBDA_R1   = 0.5            # regularizare D, nu prea mare

# ----------------- Delta (perturbația pe log-mel) -----------------
# Mai multă libertate la început + decay mai lent => G învață semnale utile fără artefacte bruște
DELTA_INIT = 0.03
DELTA_MIN  = 0.01
DELTA_DECAY = 0.9997         # per step (lent)

# ----------------- Instance noise (stabilizează D) -----------------
INST_NOISE_INIT  = 0.12
INST_NOISE_MIN   = 0.02
INST_NOISE_DECAY = 0.998     # per step (persistă mai mult)

# ----------------- Early stopping -----------------
EARLY_STOP_ENABLED  = True
EARLY_STOP_PATIENCE = 25         # epoci fără îmbunătățire
EARLY_STOP_METRIC   = "val_spec" # urmărim consistența spectrală pe val

# ----------------- Logging / Val -----------------
LOG_INTERVAL = 100
VAL_INTERVAL = 1500              # pe pași; pentru „best” robust
SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- ASVspoof (black-box evasion) -----------------
ASV_MODEL_DIR = ROOT / "ASVmodel"   # conține best_model.keras/.h5 + labels.txt
ASV_SCALER    = ASV_MODEL_DIR / "scaler.pkl"
ASV_COMBO     = "AHKMNO"
ASV_SR        = SR

# Evasion suficient de prezent, dar nu sufocant; la 1–2 pași din 2
EVASION_LAMBDA = 0.6               # direcționează G contra ASV, dar lasă loc calității
EVASION_EVERY  = 2                 # nu la fiecare pas -> mai natural
TARGET_LABEL   = "bonafide"
