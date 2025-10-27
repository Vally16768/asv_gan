from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np

# Silențiază TF și forțează CPU (ca să nu consume GPU-ul PyTorch)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from constants import ROOT
ASV_DIR = Path(ROOT) / "ASVmodel"

try:
    import tensorflow as tf
    try:
        # Ascunde toate GPU-urile din perspectiva TF
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception as e:
    raise ImportError(
        "TensorFlow/Keras nu este disponibil. Instalează-l (pip install tensorflow)."
    ) from e


class KerasASV:
    """
    Încarcă obligatoriu modelul Keras din ./ASVmodel și face scoring pe CPU.
    Adaptează automat dimensiunea vectorului de intrare (ex. 320 -> 61) prin
    interpolare liniară pe axa de features.

    Structura așteptată:
      ./ASVmodel/
        - best_model.keras sau best_model.h5   (obligatoriu)
        - labels.txt                           (opțional, pt. index bona_fide)
        - scaler.pkl                           (opțional)
        - feature_order.txt                    (opțional, informativ)
    """
    def __init__(self, loader_fn=None):
        if loader_fn is not None:
            self.model = loader_fn()
        else:
            self.model = self._auto_load_model()

        # input dim din model (ex. 61)
        self.input_dim = int(self.model.input_shape[-1])

        self.scaler = self._maybe_load_scaler()
        self.bona_index = self._infer_bona_index()
        self.feature_order = self._maybe_load_feature_order()

    # ---------- loaders ----------
    def _auto_load_model(self):
        if not ASV_DIR.exists():
            raise FileNotFoundError(f"[ASVmodel] Director inexistent: {ASV_DIR}")
        model_path = None
        if (ASV_DIR / "best_model.keras").exists():
            model_path = ASV_DIR / "best_model.keras"
        elif (ASV_DIR / "best_model.h5").exists():
            model_path = ASV_DIR / "best_model.h5"
        else:
            raise FileNotFoundError(
                f"[ASVmodel] Lipsă model: {ASV_DIR/'best_model.keras'} sau {ASV_DIR/'best_model.h5'}."
            )
        return tf.keras.models.load_model(str(model_path))

    def _maybe_load_scaler(self):
        p = ASV_DIR / "scaler.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _infer_bona_index(self) -> int:
        idx = 1  # default
        p = ASV_DIR / "labels.txt"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    labels = [ln.strip() for ln in f if ln.strip()]
                for i, name in enumerate(labels):
                    if name.lower() in ("bona_fide", "bonafide", "genuine", "real"):
                        idx = i
                        break
            except Exception:
                pass
        return idx

    def _maybe_load_feature_order(self):
        p = ASV_DIR / "feature_order.txt"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return [t.strip() for t in f if t.strip()]
            except Exception:
                pass
        return None

    # ---------- preprocessing ----------
    @staticmethod
    def _interp_1d(vec: np.ndarray, out_dim: int) -> np.ndarray:
        """Interpolează un vector 1D la dimensiunea out_dim."""
        D = vec.shape[0]
        if D == out_dim:
            return vec
        old = np.linspace(0.0, 1.0, num=D, dtype=np.float32)
        new = np.linspace(0.0, 1.0, num=out_dim, dtype=np.float32)
        return np.interp(new, old, vec).astype(np.float32)

    def _resize_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: [B, Din] -> [B, self.input_dim] prin interpolare liniară pe axa de features.
        Dacă ai nevoie de un mapping specific (după feature_order.txt),
        putem implementa un selector; momentan folosim o re-eșantionare robustă.
        """
        B, Din = X.shape
        if Din == self.input_dim:
            return X.astype(np.float32, copy=False)
        out = np.zeros((B, self.input_dim), dtype=np.float32)
        for i in range(B):
            out[i] = self._interp_1d(X[i], self.input_dim)
        return out

    def _prepare(self, feats_np: np.ndarray) -> np.ndarray:
        # 1) adaptează dimensiunea la ce așteaptă modelul Keras
        X = self._resize_features(feats_np)
        # 2) (opțional) scaler din .pkl
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                # dacă scalerul nu se potrivește, continuăm fără să oprim antrenarea
                pass
        return X

    # ---------- inference ----------
    def predict_prob_bonafide(self, feats_np: np.ndarray) -> np.ndarray:
        """
        feats_np: [B, D] (ex. concat(mean, std) din mel) — orice D.
        Returnează: [B] probabilități bona_fide.
        """
        if self.model is None:
            raise RuntimeError("[ASVmodel] Modelul Keras nu este încărcat.")
        X = self._prepare(feats_np)
        preds = self.model.predict(X, verbose=0)
        p = np.array(preds)

        if p.ndim == 2:
            # logits/prob multiclasă
            ex = np.exp(p - p.max(axis=1, keepdims=True))
            prob = ex / ex.sum(axis=1, keepdims=True)
            idx = min(self.bona_index, prob.shape[1] - 1)
            return prob[:, idx].astype(np.float32)

        # deja [B] probabilități
        return p.reshape(-1).astype(np.float32)
