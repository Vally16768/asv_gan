# dataset.py
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from constants import (
    TARGET_SR, N_MELS, N_FFT, WIN_LENGTH, HOP_LENGTH, MAX_SEC
)

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a"}

def _safe_load_with_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    import soundfile as sf
    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 1:
        data = data[None, :]                # [1, T]
    else:
        # soundfile dă de obicei [T, C]; transpune la [C, T] dacă e cazul
        if data.shape[0] < data.shape[1]:
            data = data.T
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    wav = torch.from_numpy(data)            # [C, T]
    return wav, sr

def _load_wav(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Încarcă audio fără a depinde de TorchCodec. Întoarce [1, T] float32 în [-1,1] @ target_sr.
    """
    try:
        wav, orig_sr = _safe_load_with_soundfile(path)
    except Exception:
        wav, orig_sr = torchaudio.load(path)  # [C, T]
    # mono
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    # normalize safe
    maxv = wav.abs().amax()
    if torch.isfinite(maxv) and maxv > 1.0:
        wav = wav / (maxv + 1e-8)
    return wav.contiguous().float()  # [1, T]

def _crop_or_pad(wav: torch.Tensor, max_sec: float, sr: int) -> torch.Tensor:
    """
    Cropează sau face pad la lungime fixă (max_sec). wav: [1, T]
    """
    T_target = int(max_sec * sr)
    T_cur = wav.size(-1)
    if T_cur == T_target:
        return wav
    if T_cur > T_target:
        # crop aleator
        start = np.random.randint(0, T_cur - T_target + 1)
        return wav[..., start:start+T_target]
    # pad la dreapta
    pad = T_target - T_cur
    return torch.nn.functional.pad(wav, (0, pad))

@dataclass
class Item:
    path: str

class BonaFideDataset(torch.utils.data.Dataset):
    """
    Scanează `root_dir` pentru fișiere audio bona-fide și întoarce dict:
      {
        "mel":  [N_MELS, Tm],
        "wave": [1, Tw]
      }
    Unde Tw = MAX_SEC * TARGET_SR (după crop/pad).
    """
    def __init__(self, root_dir: str, max_sec: float = MAX_SEC, shuffle: bool = True):
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"{root_dir} nu există")
        self.items: List[Item] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                self.items.append(Item(str(p)))
        if not self.items:
            raise RuntimeError(f"N-am găsit fișiere audio în {root_dir}")
        if shuffle:
            random.shuffle(self.items)

        self.max_sec = max_sec
        # MelSpectrogram + to_dB
        self.mel_tf = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=True,
            power=2.0,
            f_min=0.0,
            f_max=None,
            norm="slaney",
            mel_scale="htk",
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.items[idx].path
        wav = _load_wav(path)                              # [1, T]
        wav = _crop_or_pad(wav, self.max_sec, TARGET_SR)   # [1, Tw]

        mel = self.mel_tf(wav)                             # [1, n_mels, Tm]
        mel = self.to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)      # normalizare simplă
        mel = mel.squeeze(0)                               # [n_mels, Tm]

        return {"mel": mel, "wave": wav}
