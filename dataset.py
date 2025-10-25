# dataset.py
import glob
import random
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, MAX_SEC

_A_EXTS: Sequence[str] = ("wav", "flac", "mp3", "m4a")  # editează dacă vrei

_mel = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    power=1.0,
)

def _load_wav(path: str, sr: int = SR) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav  # [1, T]

class BonaFideDataset(Dataset):
    """
    Încarcă DOAR audio bona-fide dintr-un folder (implicit 'database/data').
    Nu există etichete; fiecare item are:
      - "wave": Tensor [T]
      - "mel":  Tensor [n_mels, Tm]
      - "path": str (calea fișierului)
    """
    def __init__(self, root_dir: str, max_sec: float = MAX_SEC):
        self.root_dir = Path(root_dir)
        self.max_len = int(max_sec * SR)
        self.files = []
        for ext in _A_EXTS:
            self.files += glob.glob(str(self.root_dir / "**" / f"*.{ext}"), recursive=True)
        if not self.files:
            raise RuntimeError(f"No audio found under {self.root_dir}. "
                               f"Put ONLY bona-fide files in this folder.")

    def __len__(self):
        return len(self.files)

    def _pad_truncate(self, wav: torch.Tensor) -> torch.Tensor:
        Tlen = wav.shape[-1]
        if Tlen > self.max_len:
            start = random.randint(0, Tlen - self.max_len)
            wav = wav[:, start:start + self.max_len]
        elif Tlen < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - Tlen))
        return wav

    def __getitem__(self, idx: int):
        path = self.files[idx]
        wav = _load_wav(path)                # [1, T]
        wav = self._pad_truncate(wav)        # [1, max_len]
        mel = _mel(wav).squeeze(0)           # [n_mels, Tm]
        mel = torch.log1p(mel)               # stabilizare
        return {"wave": wav.squeeze(0), "mel": mel, "path": path}
