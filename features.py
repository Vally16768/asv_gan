# features.py — ASV-style feature stack (stable, torch-only, device-safe)
from __future__ import annotations
from typing import Dict

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

from constants import (
    SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    FEATS_MEAN, FEATS_STD,
)

# ----------------- Helpers comune -----------------

def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

# MelSpectrogram cache per device (NU mai mutăm între CPU/GPU!)
_MEL_CACHE: Dict[str, torchaudio.transforms.MelSpectrogram] = {}

def _mel_key(device: torch.device) -> str:
    return f"{device.type}:{getattr(device, 'index', None)}"

def _get_mel(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    """
    Returnează o instanță MelSpectrogram *legată* de device-ul cerut.
    O singură instanță per device — elimină mutările CPU<->CUDA care duc
    la 'CUDA error: initialization error' din DataLoader workers.
    """
    key = _mel_key(device)
    m = _MEL_CACHE.get(key)
    if m is None:
        m = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            center=True,
            pad_mode="reflect",
            power=1.0,          # magnitudine
            norm="slaney",
            mel_scale="htk",
        ).to(device)
        _MEL_CACHE[key] = m
    return m

def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """
    wave: [B,1,T] float în [-1,1]
    return: log1p(mel) ∈ [B, M, Tf]
    """
    mel_tf = _get_mel(wave.device)                 # <- NU mai facem .to(...) la fiecare apel
    mel = mel_tf(wave.squeeze(1))                  # [B, M, Tf]
    mel = torch.log1p(mel).clamp(-8.0, 8.0)
    return _safe(mel)

# ----------------- Componente de features -----------------

def _mfcc_from_logmel(logmel: torch.Tensor, n_mfcc: int = 20) -> torch.Tensor:
    """
    MFCC proxy: proiecție liniară fixă peste axa Mel (evităm DCT pentru simplitate).
    Stabil și derivabil — suficient pentru discriminator.
    """
    B, M, T = logmel.shape
    device, dtype = logmel.device, logmel.dtype
    # bază deterministă pe device-ul curent (CPU în workers, CUDA în train)
    g = torch.Generator(device=device)
    g.manual_seed(0)
    W = torch.randn(M, n_mfcc, generator=g, device=device, dtype=dtype) / np.sqrt(M)
    mfcc = torch.einsum("bmt,mk->bkt", logmel, W)  # [B, K, T]
    return _safe(mfcc)

def _chroma_from_logmel(logmel: torch.Tensor, groups: int = 12) -> torch.Tensor:
    B, M, T = logmel.shape
    gsize = max(1, M // groups)
    chroma = []
    for i in range(groups):
        s = i * gsize; e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    return _safe(torch.cat(chroma, dim=1))  # [B, groups, T]

def _spectral_contrast(logmel: torch.Tensor, n_bands: int = 6) -> torch.Tensor:
    """
    Contrast simplu pe benzi (max - min) pentru fiecare bandă mel.
    """
    B, M, T = logmel.shape
    bands = []
    step = max(1, M // n_bands)
    for i in range(n_bands):
        s = i * step
        e = min(M, s + step)
        chunk = logmel[:, s:e, :]
        bands.append(chunk.max(dim=1, keepdim=True).values - chunk.min(dim=1, keepdim=True).values)
    return _safe(torch.cat(bands, dim=1))  # [B, n_bands, T]

def _temporal_stats(logmel: torch.Tensor) -> torch.Tensor:
    """
    Statistici temporale: mean, std pe axa Mel + medie derivată temporal (Δ pe timp).
    """
    mu = logmel.mean(dim=1, keepdim=True)              # [B,1,T]
    sd = logmel.std(dim=1, keepdim=True)               # [B,1,T]
    d1 = torch.cat([torch.zeros_like(logmel[:, :, :1]),
                    logmel[:, :, 1:] - logmel[:, :, :-1]], dim=2)
    d1 = d1.mean(dim=1, keepdim=True)                  # [B,1,T]
    return _safe(torch.cat([mu, sd, d1], dim=1))       # [B,3,T]

def _wavelets_proxy(logmel: torch.Tensor) -> torch.Tensor:
    """
    Proxy wavelets: smoothing multi-scală cu conv1d (ferestre 3,5,9).
    Concatenează rezultatele pe axa canalelor.
    """
    B, M, T = logmel.shape
    outs = []
    x = logmel
    for s in (3, 5, 9):
        k = torch.ones(1, 1, s, device=x.device, dtype=x.dtype) / s
        y = F.conv1d(x.reshape(-1, 1, T), k, padding=s//2)
        outs.append(y.reshape(B, M, T))
    return _safe(torch.cat(outs, dim=1))  # [B, 3M, T]

def _pitch_from_mel_proxy(logmel: torch.Tensor) -> torch.Tensor:
    """
    'Pitch-like' fără waveform: centroid spectral normalizat în [0,1].
    """
    B, M, T = logmel.shape
    freqs = torch.linspace(0.0, 1.0, M, device=logmel.device, dtype=logmel.dtype).view(1, M, 1)
    w = torch.exp(logmel).clamp_min(1e-8)
    centroid = (w * freqs).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
    return _safe(centroid)  # [B,1,T]

# ----------------- Stivele de features -----------------

def _stack_combo_from_logmel_core(logmel: torch.Tensor) -> torch.Tensor:
    """
    Construiește combo-ul (MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets)
    pornind DOAR din logmel (fără waveform).
    """
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)             # [B,20,T]
    chroma = _chroma_from_logmel(logmel, groups=12)         # [B,12,T]
    spec_contrast = _spectral_contrast(logmel, n_bands=6)   # [B,6,T]
    temporal = _temporal_stats(logmel)                      # [B,3,T]
    pitch_like = _pitch_from_mel_proxy(logmel)              # [B,1,T]
    wavelets = _wavelets_proxy(logmel)                      # [B,3*M,T]

    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch_like, wavelets], dim=1)
    return _safe(feats)                                     # [B, C, T]

def _normalize_feats(feats: torch.Tensor) -> torch.Tensor:
    # Acceptă scalar ori vector; în practică folosim scalar din constants.
    return (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)

def stack_asv_features_from_logmel_no_wave(logmel: torch.Tensor) -> torch.Tensor:
    """
    Varianta recomandată pentru bucla adversarială (fără Griffin-Lim / waveform).
    Returnează [B, C, T].
    """
    feats = _stack_combo_from_logmel_core(logmel)
    return _normalize_feats(feats)

def stack_asv_features_from_logmel(logmel: torch.Tensor, wave: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compat: permite apel cu (logmel, wave), dar ignoră waveform pentru determinism/stabilitate.
    Returnează [B, C, T].
    """
    return stack_asv_features_from_logmel_no_wave(logmel)

def stack_asv_features(wave: torch.Tensor) -> torch.Tensor:
    """
    Pipeline complet din waveform: wave -> logmel -> combo.
    Returnează [B, C, T].
    """
    logmel = logmel_from_wave(wave)                # [B, M, T]
    feats = _stack_combo_from_logmel_core(logmel)  # [B, C, T]
    return _normalize_feats(feats)