# constants.py
from pathlib import Path

# ========== DIRECTOARE ==========
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "database" / "data" / "asvspoof2019"
LA_TRAIN = DATA_ROOT / "ASVspoof2019_LA_train"
LA_DEV   = DATA_ROOT / "ASVspoof2019_LA_dev"
LA_EVAL  = DATA_ROOT / "ASVspoof2019_LA_eval"
LA_CM_PROTOCOLS = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
LA_ASV_PROTOCOLS = DATA_ROOT / "ASVspoof2019_LA_asv_protocols"
LA_ASV_SCORES = DATA_ROOT / "ASVspoof2019_LA_asv_scores"

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "logs"
EVAL_OUT = REPO_ROOT / "eval_out"
ASVMODEL_DIR = REPO_ROOT / "ASVmodel"   # pui aici detectoarele și vocoderul

# ========== AUDIO/FEAT ==========
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_SEC = 6.0

# ========== HIFIGAN ==========
# Folosim un model TorchScript (JIT) universal, ex: hifigan_universal_v1.pt
HIFIGAN_JIT = ASVMODEL_DIR / "hifigan_universal_v1.pt"  # pune aici fișierul .pt (TorchScript)

# ========== DETECTORS (ENSEMBLE) ==========
# Listează aici 1+ detectoare surrogate (PyTorch). train/eval le va încărca pe toate.
DETECTOR_PATHS = [
    ASVMODEL_DIR / "detector_aassist.pth",
    ASVMODEL_DIR / "detector_lcnn.pth",
    # poți comenta/adauga altele
]

# ========== SPEAKER EMBEDDINGS (opțional) ==========
# Folosim speechbrain ECAPA ca embedder (se încarcă din Internet la prima rulare; opțional).
USE_SPEAKER_LOSS = False

# ========== HYPERPARAMS IMEDIATE ==========
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-4
CRITIC_ITERS = 5
GP_LAMBDA = 10.0

LAMBDA_GAN = 1.0
LAMBDA_SPEC = 10.0
LAMBDA_EVASION = 1.0
LAMBDA_SPK = 1.0  # folosit doar dacă USE_SPEAKER_LOSS=True

SAVE_EVERY = 200
