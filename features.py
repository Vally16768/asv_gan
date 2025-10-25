# features.py — extragere de features pentru ASV (MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets)
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
import pywt

# ---------------------------------------------------------
# Fallback pentru ExtractConfig (evită importuri relative)
# ---------------------------------------------------------
try:
    from config import ExtractConfig  # dacă ai un config.py separat, îl folosește
except Exception:
    from dataclasses import dataclass
    @dataclass
    class ExtractConfig:
        sampling_rate: int = 16000
        window_length_ms: float = 25.0
        n_mels: int = 64
        fmax: float = 7600.0

# ---------------------------------------------------------
# Utilitare
# ---------------------------------------------------------
def _pf(msg: str) -> None:
    print(msg, flush=True)

def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    """n_fft rotunjit la putere a lui 2; hop = n_fft/4."""
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft = max(128, 1 << (n_fft - 1).bit_length())
    hop = max(1, n_fft // 4)
    return n_fft, hop

def _normalize_int_array_to_float32(x: np.ndarray) -> np.ndarray:
    """Normalizează PCM întreg la [-1, 1]."""
    info = np.iinfo(x.dtype)
    denom = float(max(abs(info.min), info.max))
    return x.astype(np.float32) / denom

def _load_audio_strict(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    WAV -> scipy.io.wavfile; non-WAV -> librosa.load
    Conversie mono + resampling + trim silențiu.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    ext = path.suffix.lower()
    if ext == ".wav":
        sr, x = wavfile.read(str(path))  # int16/int32/float
        if x.size == 0:
            raise ValueError(f"Empty WAV: {path}")
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        if np.issubdtype(x.dtype, np.integer):
            y = _normalize_int_array_to_float32(x)
        elif np.issubdtype(x.dtype, np.floating):
            y = x.astype(np.float32, copy=False)
        else:
            y = x.astype(np.float32, copy=False)
            ma = float(np.max(np.abs(y))) or 1.0
            y /= ma
        if int(sr) != int(target_sr):
            y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
    else:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty audio: {path}")

    # taie liniștea cap-coadă
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError(f"All-silence after trim: {path}")

    return y.astype(np.float32, copy=False), int(sr)

# ---------------------------------------------------------
# CHROMA (implementare NumPy, fără chroma_* din librosa.feature)
# ---------------------------------------------------------
def _chroma_numpy(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """
    Chromagram robust:
    - energie STFT: |STFT|^2
    - bin-urile mapate la 12 clase (C=0,...,B=11) în sistem egal-temperat
    - normalizare pe coloană
    Returnează: (12, T)
    """
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, window="hann", center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # (1 + n_fft//2,)
    k_bins = np.arange(1, S.shape[0], dtype=int)  # evită DC
    if k_bins.size == 0:
        return np.zeros((12, S.shape[1]), dtype=np.float32)

    freqs_k = freqs[k_bins]
    fC = 261.625565  # C4
    with np.errstate(divide="ignore"):
        pcs = np.round(12.0 * np.log2(np.maximum(freqs_k, 1e-12) / fC)).astype(int) % 12

    chroma = np.zeros((12, S.shape[1]), dtype=np.float32)
    for pc in range(12):
        mask = (pcs == pc)
        if np.any(mask):
            chroma[pc, :] = np.sum(S[k_bins[mask], :], axis=0).astype(np.float32)

    col_sums = np.sum(chroma, axis=0, keepdims=True) + 1e-10
    chroma /= col_sums
    return chroma

# ---------------------------------------------------------
# Pitch prin autocorelație (NumPy)
# ---------------------------------------------------------
def _pitch_autocorr(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """
    Estimează f0 pe cadre prin autocorelație normalizată.
    Întoarce vector [Tframes] (NaN unde nu detectăm).
    """
    fmin = max(50.0, sr / float(n_fft) * 1.1)
    fmax = min(800.0, sr / 4.0)
    lag_min = int(max(1, sr // fmax))
    lag_max = int(min(n_fft - 1, sr // fmin))
    if lag_max <= lag_min + 1:
        return np.full(1, np.nan, dtype=np.float32)

    win = np.hanning(n_fft).astype(np.float32)
    frames: List[float] = []
    for start in range(0, len(y) - n_fft + 1, hop):
        x = y[start:start + n_fft]
        x = (x - np.mean(x)) * win
        if not np.any(np.abs(x) > 0):
            frames.append(np.nan); continue
        r = np.correlate(x, x, mode="full")[n_fft - 1:]
        r0 = r[0] if r[0] > 0 else 1.0
        r /= r0
        seg = r[lag_min:lag_max]
        if seg.size == 0:
            frames.append(np.nan); continue
        lag = lag_min + int(np.argmax(seg))
        conf = r[lag]
        if conf < 0.1:
            frames.append(np.nan)
        else:
            frames.append(float(sr / lag))

    if not frames:
        return np.full(1, np.nan, dtype=np.float32)
    return np.array(frames, dtype=np.float32)

# ---------------------------------------------------------
# Extractor detaliat (opțional, pentru analiză/CSV)
# ---------------------------------------------------------
def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    """
    Extrage features pentru un fișier (dict de statistici); util pentru analiză offline.
    """
    feats: Dict[str, float] = {}

    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    # ZCR / RMS
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["rms_mean"] = float(np.mean(rms))

    # Spectral basics
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spec_bw_mean"]       = float(np.mean(spec_bw))
    feats["spec_rolloff_mean"]  = float(np.mean(spec_rolloff))

    # Spectral contrast (medii pe benzi)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(spec_contrast, axis=1), start=1):
        feats[f"spec_contrast_mean_{i:02d}"] = float(v)

    # Chroma (NumPy)
    chroma = _chroma_numpy(y, sr, n_fft, hop)  # (12, T)
    for i, v in enumerate(np.mean(chroma, axis=1), start=1):
        feats[f"chroma_mean_{i:02d}"] = float(v)

    # MFCC (mean & std)
    fmax_safe   = float(min(cfg.fmax, (sr / 2.0) - 1.0))
    n_mels_safe = int(min(cfg.n_mels, max(8, n_fft // 4)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
                                n_mels=n_mels_safe, fmax=fmax_safe)
    feats.update({f"mfcc_mean_{i:02d}": float(v) for i, v in enumerate(np.mean(mfcc, axis=1), start=1)})
    feats.update({f"mfcc_std_{i:02d}":  float(v) for i, v in enumerate(np.std(mfcc, axis=1),  start=1)})

    # Pitch (autocorelație)
    f0_track = _pitch_autocorr(y, sr, n_fft, hop)
    feats["pitch_mean"] = float(np.nanmean(f0_track))
    feats["pitch_std"]  = float(np.nanstd(f0_track))

    # Wavelets (mean/std pe nivele)
    coeffs = pywt.wavedec(y, "db4", level=5)
    for i, c in enumerate(coeffs, start=1):
        abs_c = np.abs(c)
        feats[f"wavelet_mean_{i:02d}"] = float(np.mean(abs_c))
        feats[f"wavelet_std_{i:02d}"]  = float(np.std(abs_c))

    return feats

# ---------------------------------------------------------
# Vector fix pentru detector (ORDINEA cerută)
# MFCC (mean+std, n_mfcc=20) + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets
# ---------------------------------------------------------
def _temporal_block(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)[0]
    return np.array([zcr.mean(), rms.mean(), cent.mean(), bw.mean(), roll.mean()], dtype=np.float32)

def _mfcc_block(y: np.ndarray, sr: int, n_fft: int, hop: int,
                n_mfcc: int = 20, n_mels: int = 64, fmax: float = 7600.0) -> np.ndarray:
    fmax = float(min(fmax, (sr / 2.0) - 1.0))
    n_mels = int(min(n_mels, max(8, n_fft // 4)))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
                              n_mels=n_mels, fmax=fmax)
    return np.concatenate([mf.mean(axis=1), mf.std(axis=1)], axis=0).astype(np.float32)  # [2*n_mfcc]

def _contrast_block(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    return sc.mean(axis=1).astype(np.float32)  # de obicei 7 valori

def vector_from_wave(y: np.ndarray, sr: int, cfg: ExtractConfig | None = None) -> np.ndarray:
    """
    Returnează vectorul de features în ORDINEA cerută:
    MFCC (mean+std, n_mfcc=20) + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets.
    """
    if cfg is None:
        cfg = ExtractConfig()
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    v_mfcc   = _mfcc_block(y, sr, n_fft, hop, n_mfcc=20, n_mels=cfg.n_mels, fmax=cfg.fmax)               # 40
    v_chroma = _chroma_numpy(y, sr, n_fft, hop).mean(axis=1).astype(np.float32)                           # 12
    v_contr  = _contrast_block(y, sr, n_fft, hop)                                                         # ~7
    v_temp   = _temporal_block(y, sr, n_fft, hop)                                                         # 5
    v_pitchf = _pitch_autocorr(y, sr, n_fft, hop)                                                         # [Tframes]
    v_pitch  = np.array([np.nanmean(v_pitchf), np.nanstd(v_pitchf)], dtype=np.float32)                    # 2
    coeffs   = pywt.wavedec(y, "db4", level=5)
    v_wl_m   = np.array([np.mean(np.abs(c)) for c in coeffs], dtype=np.float32)                           # 6
    v_wl_s   = np.array([np.std (np.abs(c)) for c in coeffs], dtype=np.float32)                           # 6
    v_wl     = np.concatenate([v_wl_m, v_wl_s], axis=0).astype(np.float32)                                # 12

    feats = np.concatenate([v_mfcc, v_chroma, v_contr, v_temp, v_pitch, v_wl], axis=0).astype(np.float32)
    return feats  # [F]

def batch_from_waves(waves: List[np.ndarray], sr: int = 16000, cfg: ExtractConfig | None = None) -> np.ndarray:
    X = [vector_from_wave(w.astype(np.float32, copy=False), sr=sr, cfg=cfg) for w in waves]
    return np.stack(X, axis=0).astype(np.float32)  # [B, F]
