from __future__ import annotations
import math
import torch
import torchaudio
import torch.nn.functional as F

from constants import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FEATS_MEAN, FEATS_STD

# valori-limită pentru stabilitate (log-mel tipic ~ [-5, +5] după log1p)
_CLIP_MIN = -8.0
_CLIP_MAX =  8.0

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    power=1.0,      # 1.0 (magnitude) e mai stabil decât 2.0 la log1p
)

def _safe(x: torch.Tensor) -> torch.Tensor:
    # înlocuiește NaN/Inf cu 0, apoi decupează
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.clamp(_CLIP_MIN, _CLIP_MAX)

def _logmel(wave: torch.Tensor) -> torch.Tensor:
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    mel = _mel(wave)                # [B,M,Tm] >= 0
    mel = torch.log1p(mel)          # stabil numeric
    return _safe(mel)

def _mfcc_from_logmel(logmel: torch.Tensor, n_mfcc: int = 20) -> torch.Tensor:
    B, M, T = logmel.shape
    device = logmel.device
    dtype = logmel.dtype
    m = torch.arange(M, device=device, dtype=dtype)
    k = torch.arange(M, device=device, dtype=dtype).unsqueeze(1)
    scale = torch.sqrt(torch.tensor(2.0 / M, device=device, dtype=dtype))
    dct = scale * torch.cos((math.pi / M) * (m + 0.5) * k)
    dct[0] = dct[0] / math.sqrt(2.0)

    x = logmel.transpose(1, 2)            # [B,T,M]
    mfcc = x @ dct.t()                    # [B,T,M]
    mfcc = mfcc[:, :, :n_mfcc].transpose(1, 2)  # [B,n_mfcc,T]
    return _safe(mfcc)

def _temporal_stats(logmel: torch.Tensor) -> torch.Tensor:
    dx = logmel[:, :, 1:] - logmel[:, :, :-1]
    dx = F.pad(dx, (1, 0))
    energy = logmel.mean(dim=1, keepdim=True)
    out = torch.cat([dx, energy], dim=1)
    return _safe(out)

def _spectral_contrast(logmel: torch.Tensor, n_bands: int = 6) -> torch.Tensor:
    B, M, T = logmel.shape
    k = max(1, M // n_bands)
    bands = []
    for i in range(0, M, k):
        band = logmel[:, i:i+k, :].clamp(min=1e-6)
        bands.append(band.max(dim=1, keepdim=True).values - band.min(dim=1, keepdim=True).values)
    out = torch.cat(bands, dim=1)
    return _safe(out)

# add at top if not already imported
from constants import HOP_LENGTH

def _pitch_proxy(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Returnează un canal de "pitch-like" la rezoluția cadrelor Mel: [B,1,target_len].
    Calculează autocorelații simple pe semnalul în eșantioane, apoi reduce la cadre prin
    average-pool cu stride=HOP_LENGTH și ajustează la exact target_len.
    """
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)   # [B,1,T_samples]
    B, _, T_s = wave.shape

    # Autocorelatii pe câteva lag-uri (proxy pt. pitch); toate la rezoluție de eșantion.
    lags = [int(0.003*SR), int(0.006*SR), int(0.01*SR), int(0.02*SR)]
    outs = []
    for lag in lags:
        if T_s <= lag:
            corr = torch.zeros(B, 1, 1, device=wave.device, dtype=wave.dtype)
        else:
            a = wave[:, :, :-lag]
            b = wave[:, :, lag:]
            corr = (a * b).mean(dim=2, keepdim=True)   # [B,1,1]
        outs.append(corr)
    pitch_samp = torch.cat(outs, dim=2)               # [B,1,4]

    # Extinde pe axa timp pentru a avea un semnal 1D ce putem pool-ui.
    # Folosim nearest pentru a evita overshoot; oricum urmează pool.
    pitch_samp = torch.nn.functional.interpolate(
        pitch_samp, size=T_s, mode="nearest"
    )                                                # [B,1,T_samples]

    # Downsample la rezoluție de cadre (aprox T_s / HOP_LENGTH).
    # Padding la dreapta pentru multiplu exact.
    pad = (0, (HOP_LENGTH - (T_s % HOP_LENGTH)) % HOP_LENGTH)
    pitch_padded = torch.nn.functional.pad(pitch_samp, pad)  # [B,1,T_pad]
    pitch_frames = torch.nn.functional.avg_pool1d(
        pitch_padded, kernel_size=HOP_LENGTH, stride=HOP_LENGTH
    )                                                        # [B,1,~T_frames]

    # Ajustează exact la target_len (mel frames)
    pitch = torch.nn.functional.interpolate(
        pitch_frames, size=target_len, mode="linear", align_corners=False
    )                                                        # [B,1,target_len]

    return _safe(pitch)

def stack_asv_features(wave: torch.Tensor) -> torch.Tensor:
    """
    Ordine: MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets
    Returnează [B, C, T_frames]
    """
    logmel = _logmel(wave)                        # [B,M,Tf]
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)   # [B,20,Tf]

    B, M, Tf = logmel.shape
    groups = 12
    gsize = max(1, M // groups)
    chroma = []
    for i in range(groups):
        s = i * gsize
        e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    chroma = torch.cat(chroma, dim=1)             # [B,12,Tf]
    chroma = _safe(chroma)

    spec_contrast = _spectral_contrast(logmel, n_bands=6)  # [B,6,Tf]
    temporal = _temporal_stats(logmel)                     # [B,M+1,Tf]

    # ⬇⬇ FIX: pitch la rezoluția cadrelor mel
    pitch = _pitch_proxy(wave, target_len=Tf)              # [B,1,Tf]

    wavelets = _wavelets_proxy(logmel)                     # [B,M*4,Tf]

    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch, wavelets], dim=1)
    feats = _safe(feats)
    feats = (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)
    return feats

def _wavelets_proxy(logmel: torch.Tensor) -> torch.Tensor:
    x = logmel
    outs = [x]
    for s in [3, 5, 9]:
        k = torch.ones(1, 1, s, device=x.device, dtype=x.dtype) / s
        y = F.conv1d(x.reshape(-1, 1, x.shape[-1]), k, padding=s//2)
        outs.append(y.reshape_as(x))
    out = torch.cat(outs, dim=1)  # [B, M*4, T]
    return _safe(out)