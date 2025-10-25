# constants.py
from __future__ import annotations
from pathlib import Path

# -------- Audio / Features --------
TARGET_SR: int = 16000
N_FFT: int = 1024
WIN_LENGTH: int = 400       # 25 ms @16k
HOP_LENGTH: int = 160       # 10 ms @16k
N_MELS: int = 64
N_MFCC: int = 20
MAX_SEC: float = 4.0        # crop/pad la ~4s

# Ordinea convenită a feature-urilor pentru detectorul ASV Keras
FEATURE_ORDER = [
    "mfcc",
    "chroma",
    "spectral_contrast",
    "temporal",
    "pitch",
    "wavelets",
]

# -------- Training defaults --------
EPOCHS: int = 3
BATCH_SIZE: int = 8
LR: float = 2e-4
CRITIC_ITERS: int = 5
GP_LAMBDA: float = 10.0
LAMBDA_GAN: float = 1.0
LAMBDA_SPEC: float = 10.0
LAMBDA_EVASION: float = 1.0
LAMBDA_SPK: float = 0.5              # pierdere de consistență speaker (opțional)

# -------- I/O --------
CHECKPOINT_DIR: str = "checkpoints"
LOG_DIR: str = "logs"

# -------- Vocoder / Detector --------
# Dacă ai un HiFi-GAN jit, pune-l aici. Dacă nu există, training-ul continuă fără vocoder.
HIFIGAN_JIT: str = str(Path("pretrained/hifigan.jit").resolve())
# Listează aici detectoarele pentru evasion loss (ex: modelul Keras)
DETECTOR_PATHS: list[str] = [str(Path("ASVmodel/best_model.keras").resolve())]
USE_SPEAKER_LOSS: bool = False  # setează True dacă vrei speaker loss cu speechbrain
