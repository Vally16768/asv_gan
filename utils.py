# utils.py
import torch, torchaudio, numpy as np, soundfile as sf
from pathlib import Path
from typing import List
from constants import SR, N_FFT, HOP_LENGTH, WIN_LENGTH

def save_wav(path, wav, sr=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    sf.write(str(path), wav, sr)

# -------- HiFi-GAN TorchScript (rapid și simplu) --------
class HiFiGANVocoder:
    """
    Așteaptă un .pt (TorchScript) care primește mel [B, n_mels, T] (log-mel sau mel?),
    și returnează waveform [B, L]. Folosim mel 'lin' (nu log1p). Vom face expm1 înainte.
    """
    def __init__(self, jit_path: Path, device):
        self.device = device
        self.ok = False
        if jit_path is None or not Path(jit_path).exists():
            print("[HiFi-GAN] TorchScript file not found, falling back to Griffin-Lim.")
            return
        self.model = torch.jit.load(str(jit_path), map_location=device)
        self.model.eval()
        self.ok = True

    @torch.inference_mode()
    def __call__(self, mel_log: torch.Tensor) -> torch.Tensor:
        # mel_log e log1p(mel); invertim
        mel = torch.expm1(mel_log).to(self.device)  # [B, n_mels, T]
        if self.ok:
            wav = self.model(mel)  # [B, L]
            return wav
        # fallback: Griffin-Lim (mai lent)
        import librosa
        mel_np = mel.detach().cpu().numpy()
        inv_mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=mel_np.shape[1])
        inv_mel = np.linalg.pinv(inv_mel)
        waves = []
        for m in mel_np:
            S = np.maximum(1e-8, inv_mel @ m)
            w = librosa.griffinlim(S, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_iter=32)
            waves.append(w.astype(np.float32))
        maxL = max(map(len, waves))
        out = np.zeros((len(waves), maxL), dtype=np.float32)
        for i,w in enumerate(waves): out[i,:len(w)] = w
        return torch.from_numpy(out)

# -------- DETECTORS (ensemble) --------
def load_detectors(paths: List[Path], device):
    """
    Fiecare model trebuie să accepte waveform [B, L] SAU mel [B, n_mels, T]
    și să returneze scor scalar per item (spoof prob / logit).
    """
    detectors = []
    for p in paths:
        p = Path(p)
        if not p.exists(): 
            print(f"[detector] WARNING: not found: {p}")
            continue
        m = torch.load(str(p), map_location=device)
        m.to(device).eval()
        detectors.append(m)
    if not detectors:
        print("[detector] No detectors found! Evasion loss will be zero.")
    return detectors

@torch.inference_mode()
def run_ensemble_detectors(detectors, wave=None, mel=None):
    """
    Rulează toate detectoarele disponibile; întoarce listă de scoruri tensor [B].
    Încearcă întâi cu wave, dacă dă eroare încearcă cu mel.
    """
    scores = []
    for det in detectors:
        try:
            s = det(wave) if wave is not None else det(mel)
        except Exception:
            s = det(mel) if mel is not None else det(wave)
        if s.dim() > 1: s = s.squeeze()
        scores.append(s)
    return scores
