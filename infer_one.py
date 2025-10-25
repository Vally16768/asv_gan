# infer_one.py — consistent with train.py (mel-in Generator), robust ckpt loader
from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

from constants import (
    ROOT, DATA_DIR, SAVE_DIR, SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH
)
from models import Generator

# ----------------- Mel pipeline (same as train.py) -----------------
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    center=True,
    pad_mode="reflect",
    power=1.0,        # magnitude
    norm="slaney",
    mel_scale="htk",
)

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

@torch.no_grad()
def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """
    wave: [B, 1, T] float32 in [-1,1]
    return: log1p(mel) în [B, N_MELS, T']
    """
    mel = _mel.to(wave.device)(wave.squeeze(1))  # [B, M, T]
    mel = torch.log1p(mel).clamp(-8.0, 8.0)
    return _safe(mel)

# ----------------- Mel->Wave reconstructor (same as train.py) -----------------
class MelToWave(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SR,
        )
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            power=1.0,
            n_iter=80,
        )

    @torch.no_grad()
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        self.inv_mel = self.inv_mel.to(logmel.device)
        self.griffin = self.griffin.to(logmel.device)
        mel = torch.expm1(logmel).clamp(min=0.0)
        spec = self.inv_mel(mel)          # [B, n_fft//2+1, T]
        wave = self.griffin(spec)         # [B, T]
        return wave

# ----------------- Utilities -----------------
def pick_random_audio(data_dir: Path) -> Path:
    exts = {".wav", ".flac", ".mp3", ".m4a"}
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"Nu am găsit fișiere audio în {data_dir}")
    return random.choice(files)

def find_latest_ema_ckpt() -> Path:
    # Preferă EMA; cade pe G_train_* dacă nu există EMA
    ema = sorted(SAVE_DIR.glob("D_train_best.pth"))
    if ema:
        return ema[-1]
    gtrain = sorted(SAVE_DIR.glob("G_train_*.pth"))
    if gtrain:
        return gtrain[-1]
    raise FileNotFoundError(f"Nu există checkpoint-uri în {SAVE_DIR}. Rulează train.py mai întâi.")

def load_state_dict_flex(path: Path, device: str):
    """
    Acceptă:
      - raw state_dict (mapăstr->tensor)
      - {'state_dict': ...}
      - {'model': ...}
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        # try common keys
        for k in ("state_dict", "model", "ema", "G", "generator"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # might itself be a raw state_dict (all tensors)
        # heuristic: check that values look like tensors
        vals = list(obj.values())
        if vals and all(isinstance(v, torch.Tensor) for v in vals):
            return obj
        # some checkpoints saved via torch.save(ema.shadow.state_dict())
        # already handled above; else raise
        raise KeyError(
            f"Nu am găsit un 'state_dict' valid în {path}. Chei: {list(obj.keys())}"
        )
    elif isinstance(obj, (list, tuple)):
        raise KeyError("Checkpoint-ul pare list/tuple, nu un state_dict dict.")
    else:
        # Some odd formats can still be an OrderedDict-like
        try:
            # Check it behaves like a mapping of tensors
            _ = [k for k in obj.keys()]  # type: ignore
            return obj
        except Exception:
            raise KeyError("Format necunoscut pentru checkpoint (nu pot extrage state_dict).")

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Calea către un checkpoint .pth (dacă omis, ia cel mai nou EMA).")
    parser.add_argument("--input", type=str, default=None, help="Fișier audio de test (altfel alege random din DATA_DIR).")
    parser.add_argument("--delta_scale", type=float, default=0.05, help="La fel ca în train.py (tanh * delta_scale).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Alege fișierul audio
    in_path = Path(args.input) if args.input else pick_random_audio(Path(DATA_DIR))
    print(f"[infer] input = {in_path}")

    # 2) Încarcă audio (mono, resample la SR)
    wav_np, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wave = torch.from_numpy(wav_np).view(1, -1)  # [1, T]
    if sr != SR:
        wave = torchaudio.functional.resample(wave, sr, SR)
    wave = wave.clamp(-1.0, 1.0).to(device)
    wave = wave.unsqueeze(1)  # [1,1,T] — pentru logmel funcția noastră

    # 3) Log-mel ca la antrenare
    mel = logmel_from_wave(wave)  # [1, N_MELS, T']

    # 4) Încarcă checkpoint (Generator primește N_MELS canale)
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ema_ckpt()
    print(f"[infer] ckpt  = {ckpt_path.name}")
    state_dict = load_state_dict_flex(ckpt_path, device)

    G = Generator(c_in=N_MELS).to(device)
    # strict=False permite mici diferențe de nume dacă ai schimbat codul recent
    G.load_state_dict(state_dict, strict=False)
    G.eval()

    # 5) Rulează G: mel_fake = mel + tanh(G(mel)) * delta_scale
    with torch.inference_mode():
        delta = torch.tanh(G(mel)) * float(args.delta_scale)
        mel_fake = _safe(mel + delta)

    # 6) Reconstruiește waveform
    mel2wav = MelToWave().to(device)
    wav_fake = mel2wav(mel_fake)  # [1, T_wav]
    wav_fake = wav_fake.squeeze(0).clamp(-1.0, 1.0).detach().cpu().numpy()

    # 7) Salvează input și output în rădăcina proiectului
    out_in  = Path(ROOT) / "inference_input.wav"
    out_gen = Path(ROOT) / "inference_result.wav"

    # salvează și inputul (resamplat/mono) la SR
    sf.write(str(out_in),  wave.squeeze().detach().cpu().numpy(), SR)
    sf.write(str(out_gen), wav_fake, SR)
    print(f"[infer] scris: {out_gen}")

if __name__ == "__main__":
    main()
