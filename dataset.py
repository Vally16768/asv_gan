# dataset.py
# dataset.py
import os
import glob
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

SR = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_SEC = 6.0  # pad/truncate

mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    n_mels=N_MELS, power=1.0
)

def load_wav(path, sr=SR):
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav  # shape [1, T]

class BonaFideDataset(Dataset):
    def __init__(self, root_dir, max_sec=MAX_SEC, sr=SR):
        self.files = []
        for ext in ("wav","flac","mp3"):
            self.files += glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        self.max_len = int(max_sec * sr)
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def pad_truncate(self, wav):
        # wav shape [1, T]
        T = wav.shape[-1]
        if T > self.max_len:
            start = random.randint(0, T - self.max_len)
            wav = wav[:, start:start+self.max_len]
        elif T < self.max_len:
            pad = self.max_len - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = load_wav(path, sr=self.sr)
        wav = self.pad_truncate(wav)  # [1, L]
        mel = mel_transform(wav)  # [n_mels, time]
        # log scale
        mel = torch.log1p(mel)
        return {
            "wave": wav.squeeze(0),     # [L]
            "mel": mel.squeeze(0),      # [n_mels, Tm]
            "path": path
        }
