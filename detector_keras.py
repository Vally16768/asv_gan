# detector_keras.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

from constants import ASV_MODEL_DIR

def _load_model(explicit: Optional[Path]) -> keras.Model:
    if tf is None or keras is None:
        raise RuntimeError("TensorFlow/Keras not available. Install tensorflow>=2.11 if you need eval.")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [ASV_MODEL_DIR / "best_model.keras", ASV_MODEL_DIR / "best_model.h5"]
    for c in candidates:
        if c.exists():
            return keras.models.load_model(str(c), compile=False)
    raise FileNotFoundError(f"ASV model not found. Tried: {', '.join(str(c) for c in candidates)}")
