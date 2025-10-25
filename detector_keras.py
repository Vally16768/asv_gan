# detector_keras.py
from __future__ import annotations
import numpy as np
import torch

from constants import TARGET_SR

class KerasDetector:
    """
    Încarcă un model Keras (.keras/.h5) și calculează scoruri pe features
    în ORDINEA: MFCC + Chroma + Spectral Contrast + Temporal + Pitch + Wavelets.
    """
    def __init__(self, model_path: str, device: str = "cpu", cfg: object | None = None):
        import tensorflow as tf  # asigură-te că ai TF >= 2.17.x pe Py3.12
        from tensorflow import keras
        self.model = keras.models.load_model(model_path, compile=False)
        self.device = device
        self.sr = getattr(cfg, "sample_rate", TARGET_SR) if cfg else TARGET_SR

        # API din features.py (patch-ul de mai sus)
        from features import batch_from_waves, ExtractConfig
        self._batch_from_waves = batch_from_waves
        self._cfg = ExtractConfig()  # sau folosește cfg dacă vrei să-l transmiți

        # Aflăm dimensiunea de intrare a modelului (F așteptat)
        ishape = getattr(self.model, "input_shape", None)
        self._F_expected = None
        if ishape is not None:
            # compat cu (None, F) sau (None, F1, F2) - păstrăm F total dacă e 2D
            if isinstance(ishape, (list, tuple)) and len(ishape) >= 2:
                if isinstance(ishape[1], (list, tuple)):
                    self._F_expected = int(np.prod(ishape[1]))
                else:
                    self._F_expected = int(ishape[1])

    def _prep_from_wave(self, wave: torch.Tensor) -> np.ndarray:
        """
        wave: [B, 1, T] sau [B, T]
        -> X: [B, F] compatibil cu modelul; pad/trunchiere dacă e necesar.
        """
        if wave.dim() == 3 and wave.size(1) == 1:
            wave = wave.squeeze(1)
        waves = wave.detach().cpu().numpy().astype(np.float32, copy=False)

        X = self._batch_from_waves([w for w in waves], sr=self.sr, cfg=self._cfg)  # [B, F_dyn]
        F_dyn = X.shape[1]
        if self._F_expected is not None and self._F_expected != F_dyn:
            # potrivim dimensiunea prin pad sau truncate (robust pentru evasion loss)
            F = self._F_expected
            if F_dyn < F:
                X = np.pad(X, ((0,0),(0, F - F_dyn)), mode="constant", constant_values=0.0)
            elif F_dyn > F:
                X = X[:, :F]
        return X.astype(np.float32, copy=False)

    def __call__(self, wave: torch.Tensor = None, mel: torch.Tensor = None) -> torch.Tensor:
        if wave is None:
            raise RuntimeError("KerasDetector necesită 'wave' pentru extragere de features.")
        X = self._prep_from_wave(wave)               # [B, F]
        y = self.model.predict(X, verbose=0)         # [B, 1] sau [B]
        y = np.squeeze(y).astype(np.float32)
        return torch.from_numpy(y)
