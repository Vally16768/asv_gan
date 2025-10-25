# utils.py
import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torchaudio

# Fallback opțional pentru citire audio (unele build-uri de torchaudio nu au FLAC)
try:
    import soundfile as sf  # type: ignore
    _HAS_SF = True
except Exception:
    _HAS_SF = False

# Extensii audio suportate (lowercase)
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """
    Setează seed-uri pentru reproducibilitate.
    Dacă deterministic_cuda=True, forțează cudnn să fie determinist (ușor mai lent).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]


def is_audio_file(path: str) -> bool:
    """Verifică extensia fișierului (case-insensitive)."""
    return Path(path).suffix.lower() in AUDIO_EXTS


def _info_via_torchaudio(path: str) -> Tuple[int, int]:
    """Metadate cu torchaudio: (sample_rate, num_frames)."""
    info = torchaudio.info(path)
    return int(info.sample_rate), int(info.num_frames)


def _info_via_soundfile(path: str) -> Tuple[int, int]:
    """Fallback metadate cu soundfile (dacă e instalat)."""
    if not _HAS_SF:
        raise RuntimeError("soundfile indisponibil ca fallback")
    i = sf.info(path)  # type: ignore
    return int(i.samplerate), int(i.frames)


def _safe_audio_info(path: str) -> Tuple[int, int]:
    """
    Obține (sample_rate, num_frames) cu torchaudio, cu fallback la soundfile.
    Ridică ultima excepție dacă nu se poate citi.
    """
    last_err: Optional[Exception] = None
    for reader in (_info_via_torchaudio, _info_via_soundfile):
        try:
            return reader(path)
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def duration_seconds(path: str) -> float:
    """Durata fișierului audio în secunde (float)."""
    sr, frames = _safe_audio_info(path)
    return float(frames) / float(sr) if sr > 0 else 0.0


def _gather_candidates(root: Path, use_patterns: bool) -> List[Path]:
    """
    Colectează fișiere candidate (extensii audio).
    - use_patterns=True: folosește rglob pe fiecare extensie (rapid și strict).
    - use_patterns=False: iterează toate fișierele și filtrează cu is_audio_file.
    """
    candidates: List[Path] = []
    if use_patterns:
        for ext in AUDIO_EXTS:
            candidates.extend(root.rglob(f"*{ext}"))
    else:
        for p in root.rglob("*"):
            if p.is_file() and is_audio_file(str(p)):
                candidates.append(p)

    # Elimină dubluri și normalizează
    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def scan_data_folder(
    root_dir: str,
    use_patterns: bool = True,
    verbose: bool = False,
    save_csv: Optional[str] = None,
) -> List[Tuple[str, int, float]]:
    """
    Enumeră recursiv fișierele audio din `root_dir` și întoarce o listă:
        (path, sample_rate, seconds)

    Nu șterge nimic — e doar sanity-check.
    """
    root = Path(root_dir)
    if verbose:
        print(f"[info] scanning: {root.resolve()}")
        print(f"[info] exists={root.exists()} is_dir={root.is_dir()}")

    if not root.exists() or not root.is_dir():
        if verbose:
            print("[warn] Calea nu există sau nu este director.")
        return []

    candidates = _gather_candidates(root, use_patterns=use_patterns)
    if verbose:
        print(f"[info] candidate audio files: {len(candidates)}")

    out: List[Tuple[str, int, float]] = []
    for p in candidates:
        try:
            sr, frames = _safe_audio_info(str(p))
            secs = round(float(frames) / float(sr), 6) if sr > 0 else 0.0
            out.append((str(p), sr, secs))
            if verbose:
                print(f"[ok] {p} — {secs}s @ {sr}Hz")
        except Exception as e:
            if verbose:
                print(f"[warn] Could not read {p}: {e}")

    if verbose:
        print(f"[info] readable audio files: {len(out)}")
        if not out:
            print(f"[warn] No readable audio under {root_dir}")

    if save_csv:
        try:
            save_path = Path(save_csv)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "sample_rate", "seconds"])
                writer.writerows(out)
            if verbose:
                print(f"[info] CSV salvat la: {save_path.resolve()}")
        except Exception as e:
            if verbose:
                print(f"[warn] Nu am putut salva CSV la {save_csv}: {e}")

    return out


# --- Utilitare simple pentru pipeline-uri ---

def list_audio_paths(root_dir: str) -> List[str]:
    """Întoarce toate căile fișierelor audio (recursiv), fără a citi metadate."""
    root = Path(root_dir)
    return [str(p) for p in _gather_candidates(root, use_patterns=True)]


def has_audio_under(root_dir: str) -> bool:
    """True dacă există cel puțin un fișier audio sub `root_dir`."""
    root = Path(root_dir)
    for ext in AUDIO_EXTS:
        if any(root.rglob(f"*{ext}")):
            return True
    return False


# --- Audio I/O convenabil ---

def save_wav(path: str, wav_tensor: torch.Tensor, sr: int = 16000):
    """
    Salvează waveform în .wav (folosește soundfile).
    wav_tensor: 1D sau 2D (dacă 2D, salvează primul exemplu).
    """
    import soundfile as sf  # lazy import
    path = str(path)
    if isinstance(wav_tensor, torch.Tensor):
        arr = wav_tensor.detach().cpu().numpy()
    else:
        arr = np.array(wav_tensor)
    if arr.ndim == 2:
        arr = arr[0]
    sf.write(path, arr, sr)


# --- Vocoder HiFi-GAN (opțional) ---

class HiFiGANVocoder:
    """
    Wrapper minimal peste un HiFi-GAN TorchScript (.jit/.pt) care primește mel [B, n_mels, T]
    și întoarce waveform [B, L]. Dacă nu ai modelul, înlocuiește-l în codul de train/eval.
    """
    def __init__(self, jit_path: Optional[str], device: str = "cpu"):
        self.jit_path = str(jit_path) if jit_path else None
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = None
        if self.jit_path and Path(self.jit_path).exists():
            self.model = torch.jit.load(self.jit_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"[vocoder] HiFi-GAN JIT not found at {self.jit_path}. Vocoder calls will raise unless replaced.")

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("HiFiGAN vocoder not loaded")
        with torch.inference_mode():
            return self.model(mel.to(self.device)).detach().cpu()


# --- Detectoare ASV (Keras) ---

# Fallback pentru ExtractConfig (dacă features.py nu o definește/nu poate fi importată)
try:
    from features import ExtractConfig  # type: ignore
except Exception:
    ExtractConfig = Any  # type: ignore

# Import defensiv al wrapper-ului Keras (nu blocăm importul utils dacă lipsesc dependențe)
try:
    from detector_keras import KerasASVDetector  # type: ignore
    _HAS_KERAS_DET = True
except Exception:
    _HAS_KERAS_DET = False
    class KerasASVDetector:  # type: ignore
        pass


def load_detectors(
    detector_paths: Sequence[str],
    device: str = "cpu",
    cfg: Optional[ExtractConfig] = None
):
    """
    Încarcă o listă de detectoare. Pentru fișiere Keras (.keras/.h5/.hdf5) folosește KerasASVDetector.
    Pentru alte formate, implementează wrapper-ele aferente după nevoie.
    """
    detectors: List[Any] = []
    for p in detector_paths:
        pstr = str(p)
        if pstr.endswith((".keras", ".h5", ".hdf5")):
            if not _HAS_KERAS_DET:
                print(f"[load_detectors] detector_keras indisponibil; skip {pstr}")
                continue
            try:
                det = KerasASVDetector(model_path=pstr, cfg=cfg, device=device)  # type: ignore
                detectors.append(det)
                print(f"[load_detectors] Loaded Keras detector: {pstr}")
            except Exception as e:
                print(f"[load_detectors] Could not load Keras detector {pstr}: {e}")
        else:
            print(f"[load_detectors] Unknown detector format for {pstr} — skipping (add wrapper if needed)")
    return detectors


def run_ensemble_detectors(
    detectors: Sequence[Any],
    *,
    wave: Optional[torch.Tensor] = None,
    mel: Optional[torch.Tensor] = None,
    sr: int = 16000
) -> List[torch.Tensor]:
    """
    Rulează toate detectoarele și întoarce o listă de scoruri (torch.Tensors scalar).
    Pentru KerasASVDetector apelăm score_wave pe un numpy waveform.
    """
    out: List[torch.Tensor] = []
    for d in detectors:
        try:
            if isinstance(d, KerasASVDetector) and hasattr(d, "score_wave"):
                # Convertim torch waveform la numpy dacă e cazul
                if isinstance(wave, torch.Tensor):
                    w_np = wave.detach().cpu().numpy()
                    if w_np.ndim == 2:  # [B, L] -> ia primul exemplu
                        w_np = w_np[0]
                elif isinstance(wave, np.ndarray):
                    w_np = wave
                else:
                    raise RuntimeError("Keras detector needs waveform input (wave). Provide wav via vocoder/dataset.")
                score = d.score_wave(w_np, sr=sr)  # type: ignore
                out.append(torch.tensor(float(score)))
            elif callable(d):
                s = d(wave if wave is not None else mel)  # type: ignore
                out.append(torch.tensor(float(s)))
            else:
                print("[run_ensemble_detectors] Unknown detector object type; skipping")
        except Exception as e:
            print(f"[run_ensemble_detectors] Detector failed: {e}")
    return out
