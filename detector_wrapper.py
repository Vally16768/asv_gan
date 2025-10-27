
import torch
import numpy as np
from features import LogMel
from constants import FEATS_MEAN, FEATS_STD
from detector_keras import KerasASV

class DetectorWrapper:
    """
    Provides:
      - keras_prob(x): numpy-only, metric (no grads)
      - torch_pooled(mel): torch pooled features (for surrogate)
    """
    def __init__(self, keras_loader_fn=None):
        self.feats = LogMel()
        self.keras = KerasASV(loader_fn=keras_loader_fn)

    @torch.no_grad()
    def keras_prob(self, wave: torch.Tensor) -> np.ndarray:
        # wave: [B,T] cpu/float
        mel = self.feats(wave).cpu().numpy()  # [B,M,Tm]
        mu = mel.mean(axis=-1)
        sd = mel.std(axis=-1)
        pooled = np.concatenate([mu, sd], axis=1)  # [B,2M]
        # standardize if needed
        pooled = (pooled - FEATS_MEAN) / max(1e-6, FEATS_STD) if isinstance(FEATS_STD, (int, float)) else pooled
        return self.keras.predict_prob_bonafide(pooled)

    def torch_pooled(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B,M,Tm]
        mu = mel.mean(dim=-1)
        sd = mel.std(dim=-1)
        pooled = torch.cat([mu, sd], dim=1)  # [B,2M]
        return pooled
