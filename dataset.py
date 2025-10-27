
import torch, torchaudio, random
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
from constants import DATA_DIR, SR
import glob

def list_audio_files(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(str(p)))
    files = [f for f in files if Path(f).suffix.lower() in {".wav", ".flac"}]
    return sorted(files)

class ASVBonafideDataset(Dataset):
    """
    Bonafide-only dataset. Loads waveforms, random crops to ~3s to 6s to add variability.
    """
    def __init__(self, roots: List[str], segment_sec=(3.0, 6.0)):
        super().__init__()
        if not roots:
            roots = [str(DATA_DIR / "**" / "*.flac")]
        self.files = list_audio_files(roots)
        self.segment_sec = segment_sec
        if len(self.files) == 0:
            raise RuntimeError(f"No audio found under: {roots}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)

        wav = wav.squeeze(0)  # [T]
        # random crop
        min_s, max_s = self.segment_sec
        seg_len = int(SR * random.uniform(min_s, max_s))
        if wav.numel() < seg_len:
            pad = seg_len - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            start = random.randint(0, max(0, wav.numel() - seg_len))
            wav = wav[start:start+seg_len]
        wav = wav.clamp(-1, 1)
        return wav, path

def pad_collate(batch):
    waves, paths = zip(*batch)
    max_len = max(w.numel() for w in waves)
    padded = []
    for w in waves:
        if w.numel() < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.numel()))
        padded.append(w)
    x = torch.stack(padded, dim=0)  # [B, T]
    return x, list(paths)
