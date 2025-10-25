from pathlib import Path
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160      # 10 ms @ 16k
WIN_LENGTH = 400      # 25 ms @ 16k
FEATS_MEAN = 0.0
FEATS_STD  = 1.0
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "database" / "data"

# ----------------- Antrenare -----------------
AMP_ENABLED = False            # lăsăm pe False până stabilizăm
BATCH_SIZE = 24               # ușor mai mic => variabilitate stabilă pe batch
EPOCHS = 1
CRITIC_ITERS = 2              # 2 e suficient cu SN + R1 + instance noise

# TTUR (stabil)
LR_G = 8e-5
LR_D = 2e-4
BETA1 = 0.0
BETA2 = 0.99

# Ponderi pierderi
LAMBDA_GAN = 1.0
LAMBDA_SPEC = 2.0             # mai mult focus pe consistența spectrală
LAMBDA_R1 = 0.5               # mai mic => mai puțin „over-regularize”

# Delta (amplitudinea perturbării pe mel)
DELTA_INIT = 0.02             # start mic => mai puține artefacte la început
DELTA_MIN = 0.005             # nu coborâm sub atât
DELTA_DECAY = 0.9995          # decay per step

# Instance noise (pe intrarea lui D): σ pornește sus și coboară
INST_NOISE_INIT = 0.15
INST_NOISE_MIN = 0.02
INST_NOISE_DECAY = 0.999      # per step

# ----------------- Early stopping -----------------
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE = 12      # epoci fără îmbunătățire pe val
EARLY_STOP_METRIC = "val_spec"  # urmărim reconstrucția spectrului

# ----------------- Logging -----------------
LOG_INTERVAL = 50
VAL_INTERVAL = 1000           # validare/checkpoint mai rar => mai rapid
SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- ASVspoof (black-box evasion) -----------------
ASV_MODEL_DIR   = ROOT / "ASVmodel"          # conține best_model.keras/.h5
ASV_SCALER      = ASV_MODEL_DIR / "scaler.pkl"
ASV_COMBO       = "AHKMNO"
ASV_SR          = SR

EVASION_LAMBDA  = 0.5         # penalizare moderată, ca regularizator
EVASION_EVERY   = 1           # la fiecare pas al G
TARGET_LABEL    = "bona_fide"
