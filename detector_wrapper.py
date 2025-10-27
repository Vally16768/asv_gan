# detector_wrapper.py — ASVspoof strict integration (AHKMNO combo via asv_adapter)
import uuid
from pathlib import Path
from typing import List

import numpy as np
import torch
import soundfile as sf

from constants import ASV_COMBO, ASV_SCALER_PATH, ASV_TMP_DIR, SR
from features import LogMel
from detector_keras import KerasASV
from asv_adapter import build_keras_input_vector

class DetectorWrapper:
    """
    Provides:
      - keras_prob_strict(wave): scoruri bona_fide prin pipeline-ul STRICT (infer_ahkmno)
      - torch_pooled(mel): pooled features (pentru surrogate)
    """
    def __init__(self, keras_loader_fn=None):
        # KerasASV STRICT: fără resize/pad; verifică dimensiunea exactă
        self.keras = KerasASV(loader_fn=keras_loader_fn, strict=True)
        self.feats = LogMel()  # încă necesar pentru surrogate (mean/std pe mel)

    @torch.no_grad()
    def keras_prob(self, wave: torch.Tensor) -> np.ndarray:
        """
        Back-compat: alias pentru keras_prob_strict.
        wave: [B, T] (CPU sau GPU, float in [-1,1])
        return: [B] numpy cu p(bonafide)
        """
        return self.keras_prob_strict(wave)

    @torch.no_grad()
    def keras_prob_strict(self, wave: torch.Tensor) -> np.ndarray:
        """
        1) pentru fiecare element din batch:
           - salvează WAV 16k în dir temporar
           - construiește vectorul de input cu asv_adapter (combo + ordinea de coloane din combos.py)
        2) rulează modelul Keras STRICT (fără resize/pad)
        """
        if wave.dim() != 2:
            raise RuntimeError(f"Expected wave [B,T], got {tuple(wave.shape)}")
        wave = wave.detach().cpu().contiguous()
        B, T = wave.shape

        # 1. Scrie batch-ul ca fișiere temporare (evităm coliziuni în DDP cu UUID)
        paths: List[Path] = []
        for b in range(B):
            p = ASV_TMP_DIR / f"tmp_{uuid.uuid4().hex}.wav"
            sf.write(str(p), wave[b].numpy(), SR)
            paths.append(p)

        try:
            # 2. Construiește X per element apoi stivește la [B, D] — ordine strictă a coloanelor
            X_list = []
            for p in paths:
                X = build_keras_input_vector(
                    audio_path=p,
                    combo=ASV_COMBO,
                    sr=SR,
                    scaler_path=Path(ASV_SCALER_PATH) if ASV_SCALER_PATH else None
                )
                if X.ndim == 1:
                    X = X[None, :]
                X_list.append(X.astype(np.float32, copy=False))
            X_batch = np.concatenate(X_list, axis=0)

            # 3. Predict STRICT (fără resize, fără scaler suplimentar — deja aplicat în adaptor dacă există)
            probs = self.keras.predict_prob_bonafide_prepared(
                X_batch, strict_dim=True, apply_scaler=False
            )
            return probs
        finally:
            # 4. Curăță fișierele temporare
            for p in paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def torch_pooled(self, mel: torch.Tensor) -> torch.Tensor:
        """ Pooled features pentru surrogate (mean|std pe axa timp). """
        mu = mel.mean(dim=-1)
        sd = mel.std(dim=-1)
        pooled = torch.cat([mu, sd], dim=1)  # [B, 2M]
        return pooled
