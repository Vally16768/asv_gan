# detector_wrapper.py — device-aware Mel everywhere
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

def wave_to_mel(wave: torch.Tensor) -> torch.Tensor:
    # acceptă [T], [1,T], [B,1,T], [B,T]
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    mel = _mel_transform.to(wave.device)(wave)  # <<< fix device
    mel = torch.log1p(torch.clamp(mel, min=0.0))
    return mel

class DetectorWrapper(torch.nn.Module):
    """
    Wrapper generic pentru detectoare PyTorch (nn.Module or TorchScript).
    Acceptă:
      - model: torch.nn.Module / ScriptModule ce primește mel sau waveform și returnează scoruri
      - model: None -> returnează zeros
    """
    def __init__(self, model: torch.nn.Module | None = None):
        super().__init__()
        self.model = model
        if self.model is not None:
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
        mel = _mel_transform.to(wave.device)(wave)  # <<< fix device
        mel = torch.log1p(torch.clamp(mel, min=0.0))
        return mel

    def forward(self, x: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        Acceptă waveform [B,1,T] sau mel [B,M,T] ori [T]/[B,T].
        Dacă self.model este None -> întoarce zero-uri.
        """
        if torch.is_tensor(x):
            if x.dim() == 3 and x.size(1) == N_MELS:
                mel = x
            elif x.dim() == 2:
                # [B,T] posibil waveform -> fă mel
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

        try:
            out = self.model(mel)
            return out.squeeze()
        except Exception as e:
            # fallback: poate modelul așteaptă waveform [B,1,T]
            try:
                wav_like = x if (torch.is_tensor(x) and x.dim() == 3 and x.size(1) == 1) else None
                if wav_like is not None:
                    out = self.model(wav_like)
                    return out.squeeze()
            except Exception:
                pass
            raise e
