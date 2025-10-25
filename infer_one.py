# infer_one.py
from __future__ import annotations
import argparse, random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

from constants import (
    ROOT, DATA_DIR, SAVE_DIR, SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH
)
from models import Generator
from features import stack_asv_features  # păstrează ordinea canalelor ca la train 

# -- Reconstrucție audio din log-mel (magnitudine) cu InverseMelScale + GriffinLim
class MelToWave:
    def __init__(self):
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SR
        )
        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            power=1.0,  # interpretăm „magnitudine”
            n_iter=60
        )

    @torch.no_grad()
    def __call__(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        logmel: [B, N_MELS, T] în scala log1p (ca în pipeline)
        return: waveform [B, T_wav] (float32, ~[-1,1])
        """
        mel_mag = torch.expm1(logmel).clamp(min=0.0)             # invers la log1p
        lin_mag = self.inv_mel(mel_mag)                          # [B, n_stft, T]
        wavs = []
        for i in range(lin_mag.size(0)):
            w = self.griffin(lin_mag[i])                         # [T_wav]
            wavs.append(w.unsqueeze(0))
        return torch.cat(wavs, dim=0)

def pick_random_audio(data_dir: Path) -> Path:
    exts = {".wav", ".flac", ".mp3", ".m4a"}
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"Nu am găsit fișiere audio în {data_dir}")
    return random.choice(files)

def load_latest_checkpoint() -> Path:
    ckpts = sorted(SAVE_DIR.glob("G_ema_ep*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Nu există checkpoint-uri în {SAVE_DIR}. Rulează train.py mai întâi.")
    return ckpts[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Calea către un checkpoint .pt (altfel ia ultimul).")
    parser.add_argument("--input", type=str, default=None, help="Fișier audio de test (altfel alege random din database/data).")
    parser.add_argument("--delta_scale", type=float, default=0.05, help="La fel ca în train.py (tanh * delta_scale).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- alege fișierul audio
    in_path = Path(args.input) if args.input else pick_random_audio(Path(DATA_DIR))
    print(f"[infer] input = {in_path}")

    # -- încarcă audio (mono, SR din constants) identic cu dataset.py
    import soundfile as sf
    import numpy as np
    wav_np, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wave = torch.from_numpy(wav_np).view(1, -1)  # [1, T]
    if sr != SR:
        wave = torchaudio.functional.resample(wave, sr, SR)

    # -- features ca la antrenare (ordinea canalelor e aceeași)
    feats = stack_asv_features(wave)  # [B, C, T_frames] 
    if feats.dim() == 3 and feats.size(0) == 1:
        pass
    feats = feats.to(device).float()
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)

    # -- încarcă checkpoint și instanțiază Generator cu c_in corect
    ckpt_path = Path(args.ckpt) if args.ckpt else load_latest_checkpoint()
    print(f"[infer] ckpt  = {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    c_in = int(ckpt.get("c_in", feats.size(1)))
    G = Generator(c_in=c_in).to(device)
    G.load_state_dict(ckpt["state_dict"], strict=False)  # state_dict salvat din EMA în train.py 
    G.eval()

    # -- rulează G cu același „tanh * delta_scale” ca în train.py
    with torch.inference_mode():
        delta = torch.tanh(G(feats)) * float(args.delta_scale)
        feats_fake = (feats + delta).clamp(-8.0, 8.0)

    # -- extrage log-mel din blocul „wavelets”: e ultima componentă, iar primele N_MELS canale din ea sunt exact logmel
    # Ordine canale: MFCC(20) + Chroma(12) + SpecContrast(6) + Temporal(N_MELS+1) + Pitch(1) + Wavelets(N_MELS*4)
    offset = 20 + 12 + 6 + (N_MELS + 1) + 1                 # = 20+12+6+129+1 = 168 când N_MELS=128
    logmel_fake = feats_fake[:, offset : offset + N_MELS, :] # [B, N_MELS, T]

    # -- reconstruiește waveform din logmel
    mel2wav = MelToWave()
    wav_fake = mel2wav(logmel_fake.cpu())  # [B, T_wav]
    wav_fake = wav_fake.squeeze(0).clamp(-1.0, 1.0).numpy()

    # -- salvează input și output în rădăcina proiectului
    out_in  = Path(ROOT) / "inference_input.wav"
    out_gen = Path(ROOT) / "inference_result.wav"
    sf.write(str(out_in),  wave.squeeze().cpu().numpy(), SR)
    sf.write(str(out_gen), wav_fake, SR)
    print(f"[infer] scris: {out_gen}")

if __name__ == "__main__":
    main()
