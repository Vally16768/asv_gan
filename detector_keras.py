# detector_keras.py
import tempfile
import os
from pathlib import Path
from typing import Dict, Sequence, Any, Optional, List

import numpy as np
import soundfile as sf  # writing temporary wav
import tensorflow as tf

from asvspoof.config import ExtractConfig
from asvspoof.features import extract_features_for_path

# IMPORTANT: If the Keras model was trained with features in a fixed order,
# ensure FEATURE_ORDER below matches that exact order. By default we will
# compute features and sort keys alphabetically to create a deterministic order.
# If you know the original training feature order, replace the sorting with a
# list containing that exact order.
def build_feature_vector_from_dict(feats: Dict[str, Any], feature_order: Optional[List[str]] = None) -> np.ndarray:
    if feature_order is None:
        keys = sorted(k for k in feats.keys() if not k in {"split", "file_id", "path", "label", "target"})
    else:
        keys = feature_order
    vec = np.array([float(feats[k]) for k in keys], dtype=np.float32)
    return vec, keys

class KerasASVDetector:
    """
    Wrapper for a Keras/TensorFlow model that expects a feature vector per file.
    The wrapper writes a temp WAV and uses asvspoof.features.extract_features_for_path to compute features.
    """

    def __init__(self, model_path: str, cfg: ExtractConfig = None, device: str = "cpu", feature_order: Optional[List[str]] = None):
        self.model_path = str(model_path)
        self.device = device
        self.cfg = cfg or ExtractConfig()
        self.feature_order = feature_order  # if None, alphabetical order of feature keys is used
        # load model (tf.keras)
        # ensure TF does not allocate all GPU memory if GPU present
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self._cached_feature_keys = None

    def _make_features_for_wav(self, wav_np: np.ndarray, sr: int) -> Dict[str, Any]:
        # write temp wav and use features.extract_features_for_path which expects a path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfh:
            tmp_path = tfh.name
        try:
            sf.write(tmp_path, wav_np.astype(np.float32), sr)
            # Build a tiny pandas-like index expected by extract_all_features / extract_features_for_path
            feats = extract_features_for_path(Path(tmp_path), self.cfg)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return feats

    def score_wave(self, wav_np: np.ndarray, sr: int = 16000) -> float:
        """
        Accepts waveform numpy array (1D float32, mono) and returns scalar score from the model.
        """
        feats = self._make_features_for_wav(wav_np, sr)
        vec, keys = build_feature_vector_from_dict(feats, self.feature_order)
        # cache keys if we decide to check mapping
        if self._cached_feature_keys is None:
            self._cached_feature_keys = keys
        # Model expects shape (1, dim)
        inp = np.expand_dims(vec, axis=0)
        pred = self.model.predict(inp, verbose=0)
        # flatten to scalar
        try:
            return float(np.squeeze(pred))
        except Exception:
            return float(pred[0])

    def get_feature_keys(self):
        return self._cached_feature_keys
