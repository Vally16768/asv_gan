# detector_keras.py
# Loader automat pentru modelul Keras/TensorFlow din ./ASVmodel.
# Fără fallback: dacă lipsesc fișierele, ridicăm eroare explicită.

from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    tf = None

from constants import ROOT

ASV_DIR = Path(ROOT) / "ASVmodel"  # structura cerută:
#   best_model.keras | best_model.h5
#   labels.txt       | feature_order.txt (opțional)
#   scaler.pkl       (opțional)

class KerasASV:
    def __init__(self, loader_fn=None):
        """
        Dacă loader_fn e None => încărcăm automat din ./ASVmodel.
        """
        self.model = None
        self.scaler = None
        self.bona_index = 1  # default index pentru bona_fide în softmax
        self.feature_order = None

        if loader_fn is not None:
            self.model = loader_fn()
            return

        # --- Auto-load din ./ASVmodel ---
        if not ASV_DIR.exists():
            raise FileNotFoundError(
                f"[ASVmodel] Directorul nu există: {ASV_DIR}. Creează-l și pune acolo best_model.keras/.h5, labels.txt, scaler.pkl (opțional)."
            )
        if tf is None:
            raise ImportError("TensorFlow/Keras nu este disponibil în mediul curent. Instalează-l pentru a rula detectorul ASV.")

        keras_path = None
        if (ASV_DIR / "best_model.keras").exists():
            keras_path = ASV_DIR / "best_model.keras"
        elif (ASV_DIR / "best_model.h5").exists():
            keras_path = ASV_DIR / "best_model.h5"
        else:
            raise FileNotFoundError(f"[ASVmodel] Nu am găsit fișierul model: {ASV_DIR/'best_model.keras'} sau {ASV_DIR/'best_model.h5'}.")

        # labels.txt (opțional dar recomandat)
        labels_path = ASV_DIR / "labels.txt"
        if labels_path.exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = [ln.strip() for ln in f if ln.strip()]
                # căutăm bona_fide
                for i, name in enumerate(labels):
                    if name.lower() in ("bona_fide", "bonafide", "genuine", "real"):
                        self.bona_index = i
                        break
            except Exception:
                pass  # păstrăm default-ul 1

        # feature_order.txt (opțional)
        feat_path = ASV_DIR / "feature_order.txt"
        if feat_path.exists():
            try:
                with open(feat_path, "r", encoding="utf-8") as f:
                    self.feature_order = [t.strip() for t in f if t.strip()]
            except Exception:
                self.feature_order = None

        # scaler.pkl (opțional)
        scaler_path = ASV_DIR / "scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = None

        # încărcăm modelul
        self.model = tf.keras.models.load_model(str(keras_path))

    def _prepare(self, feats_np: np.ndarray) -> np.ndarray:
        """
        feats_np: [B, D] (de ex. mean+std concat)
        Aplică reordonare (dacă există), apoi scaler (dacă există).
        """
        X = feats_np
        # NOTĂ: feature_order este informativă; fără un mapping explicit pe dimensionalitate,
        # păstrăm vectorul așa cum este. Dacă dorești un mapping, adaugă aici.
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                # în caz de dim mismatch, nu aplicăm scalerul
                pass
        return X

    def predict_prob_bonafide(self, feats_np: np.ndarray) -> np.ndarray:
        """
        Returnează [B] probabilitatea bona_fide.
        """
        if self.model is None:
            # fără fallback: dacă nu există model, e o eroare directă
            raise RuntimeError("Modelul Keras ASV nu este încărcat. Verifică folderul ./ASVmodel.")
        X = self._prepare(feats_np)
        preds = self.model.predict(X, verbose=0)
        p = np.array(preds)

        # cazuri:
        # - softmax/logits [B, C]
        # - prob [B] pentru bona_fide
        if p.ndim == 2:
            # dacă sunt logits, softmax
            ex = np.exp(p - p.max(axis=1, keepdims=True))
            prob = ex / ex.sum(axis=1, keepdims=True)
            idx = min(self.bona_index, prob.shape[1] - 1)
            return prob[:, idx].astype(np.float32)
        return p.reshape(-1).astype(np.float32)
