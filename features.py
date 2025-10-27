
import torch
import torchaudio
from torch import nn
from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX, POWER

class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=POWER,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] waveform in [-1,1]
        m = self.mel(x)  # [B, n_mels, Tm]
        return torch.log(m.clamp_min(self.eps))

def safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
