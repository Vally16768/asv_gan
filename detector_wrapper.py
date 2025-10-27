# detector_wrapper.py — wrapper Keras + adaptor strict (AHKMNO)
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
    - keras_prob(wave): scoruri bona_fide prin pipeline STRICT (infer_ahkmno)
    - torch_pooled(mel): pooled features (mean/std) pentru surrogate
    """
    def __init__(self, keras_loader_fn=None):
        self.keras = KerasASV(loader_fn=keras_loader_fn, strict=True)
        self.feats = LogMel()

    @torch.no_grad()
    def keras_prob(self, wave: torch.Tensor) -> np.ndarray:
        return self.keras_prob_strict(wave)

    @torch.no_grad()
    def keras_prob_strict(self, wave: torch.Tensor) -> np.ndarray:
        """
        wave: [B, T] în [-1,1]
        Pași:
         1) scrie WAV-uri temporare (UUID) pentru batch (evită coliziuni în DDP)
         2) construiește vectorii de intrare (ordinea AHKMNO) prin adaptor
         3) rulează modelul Keras deja încărcat (CPU)
        """
        if wave.dim() != 2:
            raise RuntimeError(f"Expected wave [B,T], got {tuple(wave.shape)}")
        wave = wave.detach().cpu().contiguous()
        B, _ = wave.shape

        # 1) write temp wavs
        paths: List[Path] = []
        for b in range(B):
            p = ASV_TMP_DIR / f"tmp_{uuid.uuid4().hex}.wav"
            sf.write(str(p), wave[b].numpy(), SR)
            paths.append(p)

        try:
            # 2) build input vectors
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

            # 3) predict bona_fide probabilities
            probs = self.keras.predict_prob_bonafide_prepared(
                X_batch, strict_dim=True, apply_scaler=False
            )
            return probs
        finally:
            for p in paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def torch_pooled(self, mel: torch.Tensor) -> torch.Tensor:
        """ Pooled features (mean|std pe timp) pentru surrogate. """
        mu = mel.mean(dim=-1)
        sd = mel.std(dim=-1)
        return torch.cat([mu, sd], dim=1)  # [B, 2M]
