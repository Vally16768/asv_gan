# detector_wrapper.py
import torch
import torchaudio
from typing import Union
from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    n_mels=N_MELS, power=1.0
)

class DetectorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module = None):
        super().__init__()
        self.model = model
        if self.model is not None:
            self.model.eval()

    @staticmethod
    def wave_to_mel(wave: torch.Tensor):
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        if wave.dim() == 3 and wave.size(1) == 1:
            wave = wave.squeeze(1)
        mel = _mel_transform(wave)
        mel = torch.log1p(mel)
        return mel

    def forward(self, x: Union[torch.Tensor, list]):
        if torch.is_tensor(x):
            if x.dim() == 3 and x.size(1) == N_MELS:
                mel = x
            elif x.dim() == 2:
                # [B, L] waveform or [n_mels, T]? heuristics:
                if x.size(1) > 1000:
                    mel = self.wave_to_mel(x)
                else:
                    mel = x
            elif x.dim() == 1:
                mel = self.wave_to_mel(x)
            else:
                mel = x
        else:
            raise ValueError("Unsupported input type for DetectorWrapper.forward")

        with torch.no_grad():
            if self.model is not None:
                out = self.model(mel)
            else:
                out = torch.zeros(mel.size(0))
        if torch.is_tensor(out):
            out = out.squeeze()
        return out
