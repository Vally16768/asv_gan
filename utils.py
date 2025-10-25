# utils.py
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# torchaudio e folosit doar pentru resample; soundfile pentru I/O robust
import torchaudio

from constants import TARGET_SR

# ------------------------------
# Utilitare reproducibilitate
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------
# Vocoder HiFi-GAN (TorchScript)
# ------------------------------
class HiFiGANVocoder:
    """
    Wrapper sigur pentru un vocoder HiFi-GAN salvat ca TorchScript (.jit).
    Dacă fișierul nu există sau nu poate fi încărcat, is_ready() -> False
    și __call__ va ridica o eroare explicativă (dar train.py verifică is_ready()).
    """
    def __init__(self, jit_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.jit_path = str(jit_path)
        self._ok: bool = False
        self.expected_mels: Optional[int] = None
        self.sr: int = TARGET_SR
        self.model: Optional[torch.jit.ScriptModule] = None

        if not self.jit_path or not os.path.exists(self.jit_path):
            # Nu spamăm cu print aici; train.py deja avertizează.
            self._ok = False
            return

        try:
            self.model = torch.jit.load(self.jit_path, map_location=self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            # Unele exporturi HiFi-GAN păstrează un atribut n_mel_channels
            self.expected_mels = getattr(self.model, "n_mel_channels", None)
            self._ok = True
        except Exception as e:
            print(f"[vocoder] Could not load HiFi-GAN JIT at {self.jit_path}: {e}")
            self._ok = False
            self.model = None

    def is_ready(self) -> bool:
        return self._ok and (self.model is not None)

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, n_mels, T] -> wav: [B, 1, Tw]
        Necesită is_ready() == True.
        """
        if not self.is_ready():
            raise RuntimeError("HiFiGANVocoder not ready; JIT file missing or failed to load.")

        x = mel.to(self.device).float()  # [B, M, T]
        B, M, Tm = x.shape

        # Dacă modelul a fost antrenat cu alt număr de benzi (de ex. 80), potrivim prin interpolare pe axa frecvență.
        if self.expected_mels is not None and M != int(self.expected_mels):
            x = F.interpolate(x.unsqueeze(1), size=(int(self.expected_mels), Tm),
                              mode="bilinear", align_corners=False).squeeze(1)

        # Majoritatea exporturilor HiFi-GAN acceptă [B, n_mels, T]
        y = self.model(x)  # de obicei [B, 1, Tw] sau [B, Tw]
        if isinstance(y, (list, tuple)):
            y = y[0]
        if y.dim() == 2:
            y = y.unsqueeze(1)
        return y

# ------------------------------
# I/O audio
# ------------------------------
def save_wav(path: str, wav_tensor: torch.Tensor, sr: int = TARGET_SR):
    """
    Scrie un waveform în fișier .wav folosind soundfile.
    Acceptă [T], [1, T] sau [C, T]; salvează mono dacă C==1.
    """
    import soundfile as sf

    wav = wav_tensor.detach().cpu()
    if wav.dim() == 2 and wav.size(0) == 1:
        wav = wav.squeeze(0)
    elif wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0)  # la mono

    wav_np = wav.contiguous().float().numpy()
    sf.write(path, wav_np, sr)

# ------------------------------
# Detectoare (ASV ensemble)
# ------------------------------
@dataclass
class ExtractConfig:
    # Placeholder pentru config de extragere features (compat cu ASVspoof)
    sample_rate: int = TARGET_SR
    win_length: float = 0.025
    hop_length: float = 0.010
    n_mels: int = 64
    n_mfcc: int = 20
    # extins: MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets

def load_detectors(detector_paths: Sequence[str], device: str = "cpu", cfg: Optional[ExtractConfig] = None):
    """
    Încărcare prietenoasă a detectoarelor.
    Dacă TensorFlow/Keras lipsesc sau modelul nu se potrivește, îl sărim (nu blocăm training-ul).
    """
    detectors = []
    for p in detector_paths:
        try:
            from detector_keras import KerasDetector
            det = KerasDetector(p, device=device, cfg=cfg)
            detectors.append(det)
        except Exception as e:
            print(f"[load_detectors] detector_keras indisponibil; skip {p} ({e})")
    return detectors

@torch.inference_mode()
def run_ensemble_detectors(detectors: Sequence[object],
                           wave: Optional[torch.Tensor] = None,
                           mel: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """
    Rulează un set de detectoare și întoarce o listă de scoruri [B] ca torch.Tensor (float32, pe același device ca inputul).
    Acceptă fie `wave` [B, 1, Tw], fie `mel` [B, M, T]. Dacă un detector eșuează, îl sărim.
    """
    scores: List[torch.Tensor] = []
    B = None
    dev = None

    if wave is not None:
        assert wave.dim() == 3 and wave.size(1) == 1, "wave așteptat [B,1,T]"
        B = wave.size(0)
        dev = wave.device
    elif mel is not None:
        assert mel.dim() == 3, "mel așteptat [B,M,T]"
        B = mel.size(0)
        dev = mel.device
    else:
        return scores

    for d in detectors:
        try:
            if wave is not None:
                s = d(wave)   # încercă întâi pe wave
            else:
                s = d(mel)    # altfel pe mel
            s = torch.as_tensor(s, dtype=torch.float32, device=dev)
            # Normalizează shape: [B] sau scalar -> [B]
            if s.dim() == 0:
                s = s.expand(B)
            elif s.dim() > 1:
                s = s.view(B, -1).mean(dim=1)  # agregare dacă vine [B,1] sau altceva
            scores.append(s)
        except Exception as e:
            # Skip detectorul problematic, dar nu întrerupe training-ul
            print(f"[detector] skipping a detector due to error: {e}")
            continue

    return scores
