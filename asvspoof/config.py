# asvspoof/config.py
from dataclasses import dataclass

@dataclass
class ExtractConfig:
    sampling_rate: int = 16000
    window_length_ms: float = 25.0
    n_mels: int = 64
    fmax: float = 8000.0

