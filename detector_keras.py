
# Thin wrapper to load a Keras/TensorFlow model for ASV scoring.
# It is used as a metric and to supervise the surrogate detector (no gradients).
import numpy as np

class KerasASV:
    def __init__(self, loader_fn=None):
        """
        loader_fn: callable that returns a compiled Keras model with .predict()
        You should implement loader_fn in your environment to load your ASV model.
        """
        if loader_fn is None:
            self.model = None
        else:
            self.model = loader_fn()

    def predict_prob_bonafide(self, feats_np: np.ndarray) -> np.ndarray:
        """
        feats_np: [B, D] pooled features (mean+std concat) or logits [B, C]
        Returns: [B] prob of bona_fide.
        """
        if self.model is None:
            # Fallback heuristic: 0.0 (very strict) so training relies on surrogate & recon losses.
            return np.zeros((feats_np.shape[0],), dtype=np.float32)
        preds = self.model.predict(feats_np, verbose=0)
        p = np.array(preds)
        if p.ndim == 2 and p.shape[1] >= 2:
            # assume second column is bona_fide after softmax
            # if not, adapt mapping here
            ex = np.exp(p - p.max(axis=1, keepdims=True))
            prob = ex / ex.sum(axis=1, keepdims=True)
            return prob[:, 1].astype(np.float32)
        # if already prob
        return p.reshape(-1).astype(np.float32)
