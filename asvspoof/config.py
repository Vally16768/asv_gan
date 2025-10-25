from dataclasses import dataclass

@dataclass
class ExtractConfig:
    sampling_rate: int = 16000
    window_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mfcc: int = 128
    n_chroma: int = 12
    n_spec_contrast: int = 7
    wavelet: str = "db4"
    wavelet_levels: int = 5
