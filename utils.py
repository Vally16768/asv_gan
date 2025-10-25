# utils.py
import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

# Fallback opțional dacă torchaudio nu poate citi anumite formate (ex. FLAC pe unele build-uri)
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
    """
    Verifică extensia fișierului (case-insensitive).
    """
    return Path(path).suffix.lower() in AUDIO_EXTS


def _info_via_torchaudio(path: str):
    """
    Încearcă să citească metadate audio cu torchaudio.
    Returnează (sample_rate:int, num_frames:int) sau ridică excepție.
    """
    info = torchaudio.info(path)
    # info.num_frames poate fi 0 pentru unele containere, dar foarte rar.
    return int(info.sample_rate), int(info.num_frames)


def _info_via_soundfile(path: str):
    """
    Fallback: citește metadate cu soundfile (dacă e instalat).
    Returnează (sample_rate:int, num_frames:int) sau ridică excepție.
    """
    if not _HAS_SF:
        raise RuntimeError("soundfile indisponibil ca fallback")
    i = sf.info(path)  # type: ignore
    # soundfile.info are .samplerate și .frames
    return int(i.samplerate), int(i.frames)


def _safe_audio_info(path: str) -> Tuple[int, int]:
    """
    Obține (sample_rate, num_frames) folosind torchaudio, cu fallback la soundfile dacă este disponibil.
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
    """
    Întoarce durata fișierului audio în secunde (float).
    """
    sr, frames = _safe_audio_info(path)
    return float(frames) / float(sr) if sr > 0 else 0.0


def _gather_candidates(root: Path, use_patterns: bool) -> List[Path]:
    """
    Colectează fișiere candidate (extensii audio).
    - use_patterns=True: folosește rglob pe fiecare extensie (rapid și strict).
    - use_patterns=False: iterează toate fișierele și filtrează cu is_audio_file (mai lent).
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
    # (dublurile apar rar, dar pot apărea pe FS-uri case-insensitive)
    uniq = []
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
    Enumeră recursiv fișierele audio din `root_dir` și întoarce o listă de tuple:
        (path, sample_rate, seconds)

    Nu șterge nimic — e doar sanity-check.

    Args:
        root_dir: directorul rădăcină
        use_patterns: dacă True, caută direct pe extensii (*.<ext>) pentru viteză și acuratețe
        verbose: dacă True, afișează loguri de diagnostic
        save_csv: dacă este setat, salvează raportul la această cale CSV (coloane: path,sample_rate,seconds)

    Returns:
        list[tuple[str,int,float]]
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


# --- Utilitare convenabile pentru pipeline-uri simple ---

def list_audio_paths(root_dir: str) -> List[str]:
    """
    Întoarce toate căile (string) ale fișierelor audio din `root_dir` (recursiv), fără a citi metadate.
    """
    root = Path(root_dir)
    return [str(p) for p in _gather_candidates(root, use_patterns=True)]


def has_audio_under(root_dir: str) -> bool:
    """
    True dacă există cel puțin un fișier audio (după extensie) sub `root_dir`.
    """
    root = Path(root_dir)
    for ext in AUDIO_EXTS:
        if any(root.rglob(f"*{ext}")):
            return True
    return False
