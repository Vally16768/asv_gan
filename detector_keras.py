import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

from asvspoof.config import ExtractConfig
from features import extract_features_for_path  # folosește pipeline-ul tău

try:
    import joblib
except Exception:
    joblib = None


def _read_feature_order(order_path: Path) -> Optional[List[str]]:
    if order_path.exists():
        with order_path.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return None


class KerasASVDetector:
    """
    Keras/TensorFlow wrapper:
      - extrage feature-uri cu extract_features_for_path()
      - aplică ordinea din ASVmodel/feature_order.txt (dacă există)
      - aplică scaler.pkl (StandardScaler) dacă există
    """

    def __init__(
        self,
        model_path: str,
        cfg: Optional[ExtractConfig] = None,
        device: str = "cpu",
        feature_order: Optional[List[str]] = None,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.cfg = cfg or ExtractConfig()

        # TF mem growth (nu ocupă tot GPU)
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        # Ordinea coloanelor (prioritar: fișier manifest)
        order_from_file = _read_feature_order(Path(self.model_path).with_name("feature_order.txt"))
        self.feature_order = feature_order or order_from_file  # poate fi None -> fallback la sortare

        # Scaler opțional
        self.scaler = None
        scaler_path = Path(self.model_path).with_name("scaler.pkl")
        if joblib and scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"[detector] Loaded scaler: {scaler_path}")
            except Exception as e:
                print(f"[detector] Could not load scaler: {e}")

        self._cached_keys: Optional[List[str]] = None

    def _featurize_tmp_wav(self, wav_np: np.ndarray, sr: int) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfh:
            tmp = tfh.name
        try:
            sf.write(tmp, wav_np.astype(np.float32), sr)
            feats = extract_features_for_path(Path(tmp), self.cfg)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return feats

    def _vectorize(self, feats: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        # decide order
        keys: List[str]
        if self.feature_order is not None:
            keys = self.feature_order
            missing = [k for k in keys if k not in feats]
            if missing:
                raise KeyError(f"Missing features required by model: {missing[:10]} ...")
            vec = np.array([float(feats[k]) for k in keys], dtype=np.float32)
        else:
            # fallback: ordonare alfabetică deterministă
            keys = sorted(k for k in feats.keys() if k not in {"split", "file_id", "path", "label", "target"})
            vec = np.array([float(feats[k]) for k in keys], dtype=np.float32)

        if self.scaler is not None:
            vec = self.scaler.transform(vec[None, :])[0].astype(np.float32)

        return vec, keys

    def score_wave(self, wav_np: np.ndarray, sr: int = 16000) -> float:
        feats = self._featurize_tmp_wav(wav_np, sr)
        vec, keys = self._vectorize(feats)
        if self._cached_keys is None:
            self._cached_keys = keys
        pred = self.model.predict(vec[None, :], verbose=0)
        return float(np.squeeze(pred))

    def get_feature_keys(self) -> Optional[List[str]]:
        return self._cached_keys
