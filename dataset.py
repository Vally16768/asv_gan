# dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

from constants import DATA_DIR, SR
from features import stack_asv_features

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}

def _is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS

def _scan(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and _is_audio(p)]

def _read_audio(path: Path) -> torch.Tensor:
    """Mono waveform [1, T] la SR."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    w = torch.from_numpy(wav).view(1, -1)  # [1, T]
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w

class ASVBonafideDataset(Dataset):
    """
    Returnează:
      "wave":  [1, T]
      "feats": [C, Tf]
    """
    def __init__(self, split: str = "train", data_dir: Path | None = None, use_validation: bool = True):
        self.root = Path(data_dir) if data_dir else Path(DATA_DIR)
        files = _scan(self.root)
        if not files:
            raise RuntimeError(f"No audio files (.wav/.flac/.mp3/.m4a) found under {self.root}")

        files = sorted(files)
        if not use_validation:
            # Folosește TOT setul pentru training (fără validare)
            self.files = files
        else:
            n = len(files)
            cut = max(1, int(0.9 * n))
            self.files = files[:cut] if split == "train" else files[cut:]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        wave = _read_audio(path)           # [1, T]
        feats = stack_asv_features(wave)   # [C, Tf]
        if feats.dim() == 3 and feats.size(0) == 1:
            feats = feats.squeeze(0)       # -> [C, T]
        return {"wave": wave, "feats": feats}

def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Padding pe axa timp pentru wave și features."""
    # wave
    max_Tw = max(x["wave"].size(-1) for x in batch)
    waves = []
    for x in batch:
        w = x["wave"]
        pad = max_Tw - w.size(-1)
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        waves.append(w)
    waves = torch.stack(waves, dim=0)  # [B,1,Tw]

    # feats
    C = batch[0]["feats"].size(0)
    max_Tf = max(x["feats"].size(-1) for x in batch)
    feats = []
    for x in batch:
        f = x["feats"]
        pad = max_Tf - f.size(-1)
        if pad > 0:
            f = torch.nn.functional.pad(f, (0, pad))
        feats.append(f)
    feats = torch.stack(feats, dim=0)  # [B,C,Tf]

    return {"wave": waves, "feats": feats}
