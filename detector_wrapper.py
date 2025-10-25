# detector_wrapper.py (fixed)
from typing import Union

import torch
import torchaudio

from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    power=1.0,
)

class DetectorWrapper(torch.nn.Module):
    """
    Wrapper generic pentru detectoare PyTorch (nn.Module or TorchScript).
    It accepts either:
      - model: a torch.nn.Module / ScriptModule that takes mel or waveform and returns scores
      - model: None -> returns zeros
    For Keras detection we keep separate KerasDetector wrapper (in detector_keras.py).
    """
    def __init__(self, model: torch.nn.Module | None = None):
        super().__init__()
        self.model = model
        if self.model is not None:
            # if it's a nn.Module, ensure eval
            try:
                self.model.eval()
            except Exception:
                pass

    @staticmethod
    def wave_to_mel(wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        if wave.dim() == 3 and wave.size(1) == 1:
            wave = wave.squeeze(1)
        mel = _mel_transform(wave)
        mel = torch.log1p(mel)
        return mel

    def forward(self, x: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        Accept either waveform tensors [B,1,T] or mel [B, M, T] or raw waveform vector [T] / [B, T].
        If self.model is None -> returns zero scores.
        """
        if torch.is_tensor(x):
            if x.dim() == 3 and x.size(1) == N_MELS:
                mel = x
            elif x.dim() == 2:
                # [B, T] waveform -> transform to mel
                if x.size(1) > 1000:
                    mel = self.wave_to_mel(x)
                else:
                    mel = x
            elif x.dim() == 1:
                mel = self.wave_to_mel(x.unsqueeze(0))
            else:
                mel = x
        else:
            raise ValueError("Unsupported input type for DetectorWrapper.forward")

        if self.model is None:
            return torch.zeros(mel.size(0), device=mel.device)

        # If model is nn.Module or ScriptModule that expects mel, call directly
        try:
            out = self.model(mel)
            return out.squeeze()
        except Exception as e:
            # Try feeding waveform instead
            try:
                wav_like = x if x.dim() == 3 and x.size(1) == 1 else None
                if wav_like is not None:
                    out = self.model(wav_like)
                    return out.squeeze()
            except Exception:
                pass
            raise e
