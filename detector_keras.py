from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from constants import ROOT
except Exception:
    ROOT = Path(__file__).resolve().parent

ASV_DIR = Path(ROOT) / "ASVmodel"

# TensorFlow/Keras pe CPU
try:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception as e:
    raise ImportError(
        "TensorFlow/Keras nu este disponibil. Instalează-l (ex: `pip install tensorflow`)."
    ) from e


def _find_saved_model_dir(base: Path) -> Path | None:
    for p in base.iterdir():
        if p.is_dir() and (p / "saved_model.pb").exists():
            return p
    return None


def _first(globs: list[Path]) -> Path | None:
    for g in globs:
        if g.exists():
            return g
    return None


class KerasASV:
    """
    Loader + inferență pentru modelul Keras din ./ASVmodel.
      - STRICT: fără resize/interpolare (expectă vector exact)
      - predict_prob_bonafide_prepared(X): primește deja vectori legați de caracteristici
      - Auto-detectează: .keras, .h5, sau SavedModel
      - Acceptă ASV_MODEL_PATH (env/const) ca override
    """
    def __init__(self, loader_fn=None, strict: bool = True):
        if loader_fn is not None:
            self.model = loader_fn()
        else:
            self.model = self._auto_load_model()

        self.input_shape = self.model.input_shape
        self.input_dim = int(self.input_shape[-1])

        self.scaler = self._maybe_load_scaler()
        self.bona_index = self._infer_bona_index()
        self.feature_order = self._maybe_load_feature_order()
        self.strict = strict

    # ---------- loaders ----------
    def _auto_load_model(self):
        env_path = os.getenv("ASV_MODEL_PATH")
        const_path = None
        try:
            from constants import ASV_MODEL_PATH as CONST_MODEL_PATH
            const_path = CONST_MODEL_PATH
        except Exception:
            pass

        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        if const_path:
            candidates.append(Path(const_path))

        h5 = ASV_DIR / "best_model.h5"
        kf = ASV_DIR / "best_model.keras"
        if h5.exists():
            candidates.append(h5)
        if kf.exists():
            candidates.append(kf)

        saved = _find_saved_model_dir(ASV_DIR)
        if saved is not None:
            candidates.append(saved)

        if not candidates:
            raise FileNotFoundError(
                "Nu am găsit niciun model în ./ASVmodel (best_model.h5/.keras sau SavedModel)."
            )

        last_err = None
        for c in candidates:
            try:
                mdl = tf.keras.models.load_model(str(c))
                print(f"[KerasASV] Loaded model from: {c}")
                return mdl
            except Exception as e:
                last_err = e
                # fallback: dacă .keras pare HDF5 și există .h5, încearcă și pe acela
                try:
                    if isinstance(c, Path) and c.suffix == ".keras" and (c.with_suffix(".h5")).exists():
                        alt = c.with_suffix(".h5")
                        mdl = tf.keras.models.load_model(str(alt))
                        print(f"[KerasASV] Loaded model from (fallback .h5): {alt}")
                        return mdl
                except Exception:
                    pass
                continue

        raise RuntimeError(
            "Eșec la încărcarea modelului Keras. Verifică formatul fișierului.\n"
            f"Ultima eroare: {last_err}"
        )

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

    # ---------- infer ----------
    def predict_prob_bonafide_prepared(self, X: np.ndarray, strict_dim: bool = True, apply_scaler: bool = False) -> np.ndarray:
        """
        X: [B, D] vectori de intrare (ordonați deja corect).
        return: [B] probabilități bona_fide
        """
        if strict_dim and X.ndim != 2:
            raise RuntimeError(f"X trebuie [B, D], am primit {X.shape}")
        if self.scaler is not None and apply_scaler:
            X = self.scaler.transform(X)
        logits = self.model.predict(X, verbose=0)
        if logits.ndim == 2 and logits.shape[1] > 1:
            probs = tf.nn.softmax(logits, axis=-1).numpy()[:, self.bona_index]
        else:
            probs = tf.nn.sigmoid(logits).numpy().reshape(-1)
        return probs.astype(np.float32)
