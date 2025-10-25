# features.py
from __future__ import annotations
import math
import torch
import torchaudio
import torch.nn.functional as F

from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FEATS_MEAN, FEATS_STD

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    power=2.0,  # >=0
)

def _logmel(wave: torch.Tensor) -> torch.Tensor:
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    mel = _mel(wave)                # [B,M,Tm], non-negativ
    mel = torch.log1p(mel)          # stabil numeric
    mel = torch.nan_to_num(mel, 0.0, 1e6, -1e6)
    return mel

def _mfcc_from_logmel(logmel: torch.Tensor, n_mfcc: int = 20) -> torch.Tensor:
    B, M, T = logmel.shape
    device = logmel.device
    dtype = logmel.dtype

    m = torch.arange(M, device=device, dtype=dtype)              # [M]
    k = torch.arange(M, device=device, dtype=dtype).unsqueeze(1) # [M,1]
    scale = torch.sqrt(torch.tensor(2.0 / M, device=device, dtype=dtype))
    dct = scale * torch.cos((math.pi / M) * (m + 0.5) * k)       # [M,M]
    dct[0] = dct[0] / math.sqrt(2.0)

    x = logmel.transpose(1, 2)            # [B,T,M]
    mfcc = x @ dct.t()                    # [B,T,M]
    mfcc = mfcc[:, :, :n_mfcc].transpose(1, 2)  # [B,n_mfcc,T]
    mfcc = torch.nan_to_num(mfcc, 0.0, 1e6, -1e6)
    return mfcc

def _temporal_stats(logmel: torch.Tensor) -> torch.Tensor:
    dx = logmel[:, :, 1:] - logmel[:, :, :-1]
    dx = F.pad(dx, (1, 0))
    energy = logmel.mean(dim=1, keepdim=True)
    out = torch.cat([dx, energy], dim=1)
    return torch.nan_to_num(out, 0.0, 1e6, -1e6)

def _spectral_contrast(logmel: torch.Tensor, n_bands: int = 6) -> torch.Tensor:
    B, M, T = logmel.shape
    k = max(1, M // n_bands)
    bands = []
    for i in range(0, M, k):
        band = logmel[:, i:i+k, :].clamp(min=1e-6)
        bands.append(band.max(dim=1, keepdim=True).values - band.min(dim=1, keepdim=True).values)
    out = torch.cat(bands, dim=1)
    return torch.nan_to_num(out, 0.0, 1e6, -1e6)

def _pitch_proxy(wave: torch.Tensor) -> torch.Tensor:
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    B, _, T = wave.shape
    lags = [int(0.003*SR), int(0.006*SR), int(0.01*SR), int(0.02*SR)]
    outs = []
    for lag in lags:
        if T <= lag:  # protecție
            corr = torch.zeros(B, 1, 1, device=wave.device, dtype=wave.dtype)
        else:
            a = wave[:, :, :-lag]
            b = wave[:, :, lag:]
            corr = (a * b).mean(dim=2, keepdim=True)
        outs.append(corr)
    pitch = torch.cat(outs, dim=2)               # [B,1,4]
    pitch = torch.nan_to_num(pitch, 0.0, 1e6, -1e6)
    return pitch

def _wavelets_proxy(logmel: torch.Tensor) -> torch.Tensor:
    x = logmel
    outs = [x]
    for s in [3, 5, 9]:
        k = torch.ones(1, 1, s, device=x.device, dtype=x.dtype) / s
        y = F.conv1d(x.reshape(-1, 1, x.shape[-1]), k, padding=s//2)
        outs.append(y.reshape_as(x))
    out = torch.cat(outs, dim=1)  # [B, M*4, T]
    return torch.nan_to_num(out, 0.0, 1e6, -1e6)

def stack_asv_features(wave: torch.Tensor) -> torch.Tensor:
    """
    Ordine: MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets
    Returnează [B, C, T]
    """
    logmel = _logmel(wave)                        # [B,M,T]
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)   # [B,20,T]

    B, M, T = logmel.shape
    groups = 12
    gsize = max(1, M // groups)
    chroma = []
    for i in range(groups):
        s = i * gsize
        e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    chroma = torch.cat(chroma, dim=1)             # [B,12,T]
    chroma = torch.nan_to_num(chroma, 0.0, 1e6, -1e6)

    spec_contrast = _spectral_contrast(logmel, n_bands=6)  # [B,6,T]
    temporal = _temporal_stats(logmel)                     # [B,M+1,T]

    pitch = _pitch_proxy(wave)                             # [B,1,4]
    pitch = F.interpolate(pitch, size=T, mode="linear", align_corners=False)  # [B,1,T]

    wavelets = _wavelets_proxy(logmel)                     # [B,M*4,T]

    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch, wavelets], dim=1)
    feats = torch.nan_to_num(feats, 0.0, 1e6, -1e6)
    feats = (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)
    return feats
