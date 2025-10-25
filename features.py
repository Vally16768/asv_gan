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
    power=1.0,      # 1.0 (magnitude) + log1p => mai stabil
)

def _safe(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.clamp(_CLIP_MIN, _CLIP_MAX)

def _logmel(wave: torch.Tensor) -> torch.Tensor:
    """
    wave: [B,1,T] / [B,T] -> logmel [B,M,Tf], device-aware
    """
    if wave.dim() == 3 and wave.size(1) == 1:
        wave = wave.squeeze(1)
    mel = _mel.to(wave.device)(wave)        # <<< fix device
    mel = torch.log1p(torch.clamp(mel, min=0.0))
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

    x = logmel.transpose(1, 2)                  # [B,T,M]
    mfcc = x @ dct.t()                          # [B,T,M]
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

def _pitch_proxy(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Un canal „pitch-like” la rezoluția cadrelor Mel: [B,1,target_len].
    Aproximăm prin autocorelație simplă + average-pool pe eșantioane (stride=HOP_LENGTH),
    apoi interpolăm exact la target_len.
    """
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)   # [B,1,T_samples]
    B, _, T_s = wave.shape
    # autocorelație scurtă
    max_lag = int(0.02 * SR)  # ~20ms
    pad = (0, max_lag)
    wpad = F.pad(wave, pad)
    pitch_samp = (wpad[:, :, :-max_lag] * wpad[:, :, max_lag:]).mean(dim=1, keepdim=True)  # [B,1,T_samples]
    # downsample la rezoluție de cadre
    pad2 = (0, (HOP_LENGTH - (pitch_samp.shape[-1] % HOP_LENGTH)) % HOP_LENGTH)
    pitch_padded = F.pad(pitch_samp, pad2)
    pitch_frames = F.avg_pool1d(pitch_padded, kernel_size=HOP_LENGTH, stride=HOP_LENGTH)   # [B,1,~T]
    # ajustează fix la target_len
    pitch = F.interpolate(pitch_frames, size=target_len, mode="linear", align_corners=False)
    return _safe(pitch)

def _wavelets_proxy(logmel: torch.Tensor) -> torch.Tensor:
    x = logmel
    outs = [x]
    for s in [3, 5, 9]:
        k = torch.ones(1, 1, s, device=x.device, dtype=x.dtype) / s
        y = F.conv1d(x.reshape(-1, 1, x.shape[-1]), k, padding=s//2)
        outs.append(y.reshape_as(x))
    out = torch.cat(outs, dim=1)  # [B, M*4, T]
    return _safe(out)

def logmel_from_wave(wave: torch.Tensor) -> torch.Tensor:
    """Conveniență: log-mel [B,M,T] stabil numeric din waveform [B,1,T]."""
    return _logmel(wave)

def stack_asv_features(wave: torch.Tensor) -> torch.Tensor:
    """
    Ordine (exact cum ai cerut): MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets
    Returnează [B, C, T_frames].
    """
    logmel = _logmel(wave)                          # [B,M,Tf]
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)     # [B,20,Tf]

    B, M, Tf = logmel.shape
    groups = 12
    gsize = max(1, M // groups)
    chroma = []
    for i in range(groups):
        s = i * gsize; e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    chroma = _safe(torch.cat(chroma, dim=1))        # [B,12,Tf]

    spec_contrast = _spectral_contrast(logmel, n_bands=6)  # [B,6,Tf]
    temporal = _temporal_stats(logmel)                     # [B,M+1,Tf]
    pitch = _pitch_proxy(wave, target_len=Tf)              # [B,1,Tf]
    wavelets = _wavelets_proxy(logmel)                     # [B,M*4,Tf]

    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch, wavelets], dim=1)
    feats = _safe(feats)
    feats = (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)
    return feats

def stack_asv_features_from_logmel(logmel: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
    """
    Aceeași ordine ca mai sus, dar primește logmel deja calculat (util pentru mel_fake).
    """
    B, M, Tf = logmel.shape
    mfcc = _mfcc_from_logmel(logmel, n_mfcc=20)

    groups = 12
    gsize = max(1, M // groups)
    chroma = []
    for i in range(groups):
        s = i * gsize; e = min(M, s + gsize)
        chroma.append(logmel[:, s:e, :].mean(dim=1, keepdim=True))
    chroma = _safe(torch.cat(chroma, dim=1))

    spec_contrast = _spectral_contrast(logmel, n_bands=6)
    temporal = _temporal_stats(logmel)
    pitch = _pitch_proxy(wave, target_len=Tf)
    wavelets = _wavelets_proxy(logmel)

    feats = torch.cat([mfcc, chroma, spec_contrast, temporal, pitch, wavelets], dim=1)
    feats = _safe(feats)
    feats = (feats - float(FEATS_MEAN)) / (float(FEATS_STD) + 1e-8)
    return feats
