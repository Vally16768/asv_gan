# asvspoof/features.py — simplu, secvențial, stabil; include CHROMA & PITCH (implementare NumPy)
from pathlib import Path
from typing import Dict, List, Any, Tuple
from time import monotonic

import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
import pywt

from .config import ExtractConfig


def _pf(msg: str) -> None:
    print(msg, flush=True)


def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    """n_fft putere a lui 2, hop = n_fft/4 (automat)."""
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft = max(128, 1 << (n_fft - 1).bit_length())  # rotunjire la putere de 2, min 128
    hop = max(1, n_fft // 4)
    return n_fft, hop


def _normalize_int_array_to_float32(x: np.ndarray) -> np.ndarray:
    """Normalizează PCM întreg la [-1, 1] (fără hardcodări pe tip)."""
    info = np.iinfo(x.dtype)
    denom = float(max(abs(info.min), info.max))
    return x.astype(np.float32) / denom


def _load_audio_strict(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    .wav  -> scipy.io.wavfile (evităm libsndfile pentru stabilitate)
    altceva (ex. .flac) -> librosa.load (backend implicit)
    Conversie mono + resampling dacă e necesar + trim tăcere.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    ext = path.suffix.lower()

    if ext == ".wav":
        sr, x = wavfile.read(str(path))  # x: int16/int32/float
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
        # Non-WAV (e.g., FLAC) — librosa gestionează resampling + mono direct
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty audio: {path}")

    # Taie liniștea cap-coadă (prag ok pentru ASVspoof)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError(f"All-silence after trim: {path}")

    return y.astype(np.float32, copy=False), int(sr)


# ----------------------------
# CHROMA (implementare NumPy)
# ----------------------------
def _chroma_numpy(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """
    Chromagram robust fără librosa.feature.chroma_* (evităm potențiale segfault-uri).
    - STFT energie: |STFT|^2
    - mapăm fiecare bin de frecvență la una din cele 12 clase (C=0,...,B=11)
      folosind acordaj egal-temperat cu referință C4 = 261.625565 Hz.
    - normalizare pe coloană (sum=1) pentru robustețe la varianta de energie.

    Returnează: (12, T)
    """
    # STFT (folosim implementarea sigură din librosa, care bazat pe NumPy FFT)
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, window="hann", center=True)) ** 2
    # Frecvențe pentru fiecare bin
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # (1 + n_fft//2,)
    # Evităm DC (0 Hz)
    k_bins = np.arange(1, S.shape[0], dtype=int)
    if k_bins.size == 0:
        return np.zeros((12, S.shape[1]), dtype=np.float32)

    freqs_k = freqs[k_bins]
    # Referință pentru C (C4 ~ 261.625565 Hz)
    fC = 261.625565

    # Pitch class index pentru fiecare bin > 0 Hz
    # pc = round(12 * log2(f / fC)) mod 12
    with np.errstate(divide="ignore"):
        pcs = np.round(12.0 * np.log2(np.maximum(freqs_k, 1e-12) / fC)).astype(int) % 12

    # Agregăm energia binurilor pe clase
    chroma = np.zeros((12, S.shape[1]), dtype=np.float32)
    for pc in range(12):
        mask = (pcs == pc)
        if np.any(mask):
            chroma[pc, :] = np.sum(S[k_bins[mask], :], axis=0).astype(np.float32)

    # Normalizare pe coloană (evită dominanța energiei absolute)
    col_sums = np.sum(chroma, axis=0, keepdims=True) + 1e-10
    chroma /= col_sums
    return chroma


# ---------------------------------
# Pitch (implementare autocorelație)
# ---------------------------------
def _pitch_autocorr(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """
    Estimează f0 pe cadre prin autocorelație normalizată (fără librosa.yin).
    - fmin/fmax setate automat din (sr, n_fft).
    - întoarce vector f0 pe cadre (NaN când nu se găsește un vârf acceptabil).
    """
    # Limite rezonabile, automat:
    fmin = max(50.0, sr / float(n_fft) * 1.1)   # cel puțin 1.1 perioade în fereastră
    fmax = min(800.0, sr / 4.0)                 # limităm sus ca să fie robust

    lag_min = int(max(1, sr // fmax))
    lag_max = int(min(n_fft - 1, sr // fmin))   # <= lungimea ferestrei

    if lag_max <= lag_min + 1:
        # Cadru prea scurt pentru estimare
        return np.full(1, np.nan, dtype=np.float32)

    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, len(y) - n_fft + 1, hop):
        x = y[start:start + n_fft]
        x = x - np.mean(x)
        x = x * win
        if not np.any(np.abs(x) > 0):
            frames.append(np.nan)
            continue

        # Autocorelație "full", păstrăm partea pozitivă
        r = np.correlate(x, x, mode="full")[n_fft - 1:]
        r0 = r[0] if r[0] > 0 else 1.0
        r /= r0

        seg = r[lag_min:lag_max]
        if seg.size == 0:
            frames.append(np.nan)
            continue

        lag = lag_min + int(np.argmax(seg))
        conf = r[lag]
        # Prag mic de încredere pentru a filtra cadrele nepotrivite
        if conf < 0.1:
            frames.append(np.nan)
        else:
            f0 = float(sr / lag)
            frames.append(f0)

    if not frames:
        return np.full(1, np.nan, dtype=np.float32)

    return np.array(frames, dtype=np.float32)


def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    """
    Extrage features robuste pentru un fișier. Log granular pe etape. Fail-fast cu context clar.
    Include: zcr, rms, spectral basics, spectral contrast, chroma (NumPy), mfcc, pitch (autocorr), wavelets.
    """
    feats: Dict[str, float] = {}

    # --- LOAD ---
    _pf(f"    STAGE: load        START :: {path}")
    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    _pf(f"    STAGE: load        DONE  :: len={len(y)} sr={sr}")

    # Pregătire ferestre (automat din SR + fereastră ms)
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    # --- ZCR/RMS ---
    _pf("    STAGE: zcr_rms     START")
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["rms_mean"] = float(np.mean(rms))
    _pf("    STAGE: zcr_rms     DONE")

    # --- Spectral basic (centroid/bandwidth/rolloff) ---
    _pf("    STAGE: spectral    START")
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spec_bw_mean"]       = float(np.mean(spec_bw))
    feats["spec_rolloff_mean"]  = float(np.mean(spec_rolloff))
    _pf("    STAGE: spectral    DONE")

    # --- Spectral contrast ---
    _pf("    STAGE: contrast    START")
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(spec_contrast, axis=1), start=1):
        feats[f"spec_contrast_mean_{i:02d}"] = float(v)
    _pf("    STAGE: contrast    DONE")

    # --- Chroma (NumPy) ---
    _pf("    STAGE: chroma      START")
    chroma = _chroma_numpy(y, sr, n_fft, hop)  # (12, T)
    for i, v in enumerate(np.mean(chroma, axis=1), start=1):
        feats[f"chroma_mean_{i:02d}"] = float(v)
    _pf("    STAGE: chroma      DONE")

    # --- MFCC (auto-ajustare fmax/n_mels pentru a evita filtre goale) ---
    _pf("    STAGE: mfcc        START")
    fmax_safe   = float(min(cfg.fmax, (sr / 2.0) - 1.0))       # < Nyquist
    n_mels_safe = int(min(cfg.n_mels, max(8, n_fft // 4)))     # conservator, fără canale goale
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels_safe, fmax=fmax_safe
    )
    feats.update({f"mfcc_mean_{i:02d}": float(v) for i, v in enumerate(np.mean(mfcc, axis=1), start=1)})
    feats.update({f"mfcc_std_{i:02d}":  float(v) for i, v in enumerate(np.std(mfcc, axis=1),  start=1)})
    _pf("    STAGE: mfcc        DONE")

    # --- Pitch (autocorelație NumPy) ---
    _pf("    STAGE: pitch_acf   START")
    f0_track = _pitch_autocorr(y, sr, n_fft, hop)  # vector pe cadre (NaN unde nu detectăm)
    if np.all(~np.isfinite(f0_track)):
        raise ValueError("Pitch extraction (autocorr) failed (all NaN)")
    feats["pitch_mean"] = float(np.nanmean(f0_track))
    feats["pitch_std"]  = float(np.nanstd(f0_track))
    _pf("    STAGE: pitch_acf   DONE")

    # --- Wavelets ---
    _pf("    STAGE: wavelets    START")
    coeffs = pywt.wavedec(y, "db4", level=5)
    if not coeffs:
        raise ValueError("Wavelet decomposition failed")
    for i, c in enumerate(coeffs, start=1):
        abs_c = np.abs(c)
        feats[f"wavelet_mean_{i:02d}"] = float(np.mean(abs_c))
        feats[f"wavelet_std_{i:02d}"]  = float(np.std(abs_c))
    _pf("    STAGE: wavelets    DONE")

    return feats


def extract_all_features(df_index: pd.DataFrame, cfg: ExtractConfig, *, verbose: bool = True) -> pd.DataFrame:
    """
    Strict + secvențial:
      - preflight minimal (load + RMS)
      - parcurge toate fișierele; fail-fast cu path + motiv
      - log START/DONE per fișier și etape intermediare
    """
    jobs: List[Dict[str, Any]] = [
        {
            "split": r.split,
            "file_id": r.file_id,
            "abs_path": r.abs_path,
            "label": (r.label if isinstance(r.label, str) else None),
            "target": (int(r.target) if pd.notna(r.target) else None),
        }
        for r in df_index.itertuples(index=False)
    ]
    if not jobs:
        return pd.DataFrame(columns=["split", "file_id", "path", "label", "target"])

    # Preflight minimal
    first = jobs[0]
    p0 = Path(first["abs_path"])
    _pf(f"[*] Preflight minimal: load+RMS :: {p0}")
    y0, sr0 = _load_audio_strict(p0, cfg.sampling_rate)
    rms0 = float(np.sqrt(np.mean(y0 ** 2)))
    _pf(f"[*] Preflight OK :: len={len(y0)} sr={sr0} rms~{rms0:.4f}")

    rows: List[Dict[str, object]] = []

    for i, jd in enumerate(jobs, start=1):
        p = Path(jd["abs_path"])
        if verbose:
            _pf(f"[{i}/{len(jobs)}] START {jd['split']} {jd['file_id']} :: {p}")
        t0 = monotonic()
        try:
            feats = extract_features_for_path(p, cfg)
        except Exception as e:
            _pf(f"[!] FAIL {jd['split']} {jd['file_id']} :: {p} :: {type(e).__name__}: {e}")
            raise
        dt = monotonic() - t0
        if verbose:
            _pf(f"[{i}/{len(jobs)}] DONE  {jd['split']} {jd['file_id']} :: {p} :: {dt:.3f}s")

        base = {
            "split": jd["split"],
            "file_id": jd["file_id"],
            "path": str(p),
            "label": jd["label"],
            "target": jd["target"],
        }
        base.update(feats)
        rows.append(base)

    feat_df = pd.DataFrame(rows)
    cols_order = ["split", "file_id", "path", "label", "target"]
    other_cols = sorted([c for c in feat_df.columns if c not in cols_order])
    return feat_df[cols_order + other_cols]
