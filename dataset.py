# dataset.py
import glob
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from constants import SR, DATA_DIR, SEGMENT_SAMPLES


# ----------------- Utils -----------------
def list_audio_files(patterns: List[str]) -> List[str]:
    """
    Acceptă glob patterns (ex: str(DATA_DIR / '**' / '*.flac')) și întoarce
    toate fișierele WAV/FLAC sortate.
    """
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(str(p), recursive=True))
    files = [f for f in files if Path(f).suffix.lower() in {".wav", ".flac"}]
    return sorted(files)


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [C, T] -> [1, T] (medie pe canale dacă C > 1)
    """
    if wav.dim() != 2:
        raise RuntimeError(f"Expected [C, T], got shape={tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _resample_if_needed(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)


def _fixed_crop_or_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    x: [T] -> [target_len]
    - dacă T >= target_len: crop aleator
    - dacă T  < target_len: pad cu zerouri la dreapta
    """
    T = x.numel()
    if T == target_len:
        return x
    if T > target_len:
        start = random.randint(0, T - target_len)
        return x[start:start + target_len]
    # pad right
    pad = target_len - T
    return torch.nn.functional.pad(x, (0, pad))


# ----------------- Dataset -----------------
class ASVBonafideDataset(Dataset):
    """
    Bonafide-only dataset.
    - Încarcă WAV/FLAC, convertește la 16 kHz mono.
    - Crop/pad fix la SEGMENT_SAMPLES (control de memorie).
    - Întoarce: (waveform[T], path_str)
    """
    def __init__(self, roots: List[str] | None = None):
        super().__init__()
        if not roots or len(roots) == 0:
            # fallback: toate .flac recursion sub DATA_DIR
            roots = [str(DATA_DIR / "**" / "*.flac")]
        self.files = list_audio_files(roots)
        if len(self.files) == 0:
            raise RuntimeError(f"No audio found under: {roots}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]

        # wave: [C, T], int16/float32 în funcție de fișier
        wav, sr = torchaudio.load(path)

        # mono + resample + clamp
        wav = _to_mono(wav)                   # [1, T]
        wav = _resample_if_needed(wav, sr, SR)
        wav = wav.clamp_(-1.0, 1.0).squeeze(0)  # [T], float32

        # crop/pad fix la SEGMENT_SAMPLES
        wav = _fixed_crop_or_pad_1d(wav, SEGMENT_SAMPLES)  # [SEGMENT_SAMPLES]

        return wav, path


# ----------------- Collate -----------------
def pad_collate(batch: List[Tuple[torch.Tensor, str]]):
    """
    Primește o listă de (wav[T], path). Aliniează pe lungimea maximă,
    apoi stivuiește la [B, T]. Dacă toate au aceeași lungime (cazul nostru),
    costul este minim.
    """
    waves, paths = zip(*batch)
    max_len = max(w.numel() for w in waves)
    padded = []
    for w in waves:
        if w.numel() < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.numel()))
        padded.append(w)
    x = torch.stack(padded, dim=0)  # [B, T]
    return x, list(paths)
