from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- config & roots ---
try:
    from constants import ROOT  # and maybe ASV_MODEL_PATH if you add it
except Exception:
    ROOT = Path(__file__).resolve().parent

ASV_DIR = Path(ROOT) / "ASVmodel"

# Keras/TensorFlow on CPU
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
    # Caută un folder SavedModel (conține saved_model.pb)
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
    Noutăți:
      - mod STRICT: fără interpolare/resize; verifică dimensiunea de intrare
      - predict_prob_bonafide_prepared(X): pentru input deja pregătit (ex. asv_adapter)
      - auto-detectează modelul: .keras, .h5, sau SavedModel
      - permite cale explicită prin env ASV_MODEL_PATH sau constantă (dacă e definită în constants.py)
    """
    def __init__(self, loader_fn=None, strict: bool = True):
        # 1) Custom loader (dacă e oferit)
        if loader_fn is not None:
            self.model = loader_fn()
        else:
            self.model = self._auto_load_model()

        # input dim din model (ex. 61)
        self.input_shape = self.model.input_shape
        self.input_dim = int(self.input_shape[-1])

        self.scaler = self._maybe_load_scaler()
        self.bona_index = self._infer_bona_index()
        self.feature_order = self._maybe_load_feature_order()

        self.strict = strict  # dacă True, nu facem resize la intrare

    # ---------- loaders ----------
    def _auto_load_model(self):
        # 0) Cale explicită prin ENV sau constants.ASV_MODEL_PATH
        env_path = os.getenv("ASV_MODEL_PATH")
        const_path = None
        try:
            from constants import ASV_MODEL_PATH as CONST_MODEL_PATH  # optional
            const_path = CONST_MODEL_PATH
        except Exception:
            pass

        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        if const_path:
            candidates.append(Path(const_path))

        # 1) Preferă .h5 (compat Keras v2/v3) când coexistă cu .keras
        h5 = ASV_DIR / "best_model.h5"
        kf = ASV_DIR / "best_model.keras"
        if h5.exists():
            candidates.append(h5)
        if kf.exists():
            candidates.append(kf)

        # 2) Alte fișiere din director
        #   - întâi .h5, apoi .keras
        for p in sorted(ASV_DIR.glob("*.h5")):
            if p not in candidates:
                candidates.append(p)
        for p in sorted(ASV_DIR.glob("*.keras")):
            if p not in candidates:
                candidates.append(p)

        # 3) SavedModel directory (conține saved_model.pb)
        for p in ASV_DIR.iterdir():
            if p.is_dir() and (p / "saved_model.pb").exists():
                candidates.append(p)

        if not candidates:
            raise FileNotFoundError(
                "Nu am găsit niciun model în ./ASVmodel. Pune best_model.h5 sau setează ASV_MODEL_PATH."
            )

        last_err = None
        for c in candidates:
            try:
                mdl = tf.keras.models.load_model(str(c))
                print(f"[KerasASV] Loaded model from: {c}")
                return mdl
            except Exception as e:
                last_err = e
                # fallback special: dacă extensia e .keras dar pare HDF5, încearcă din nou ca HDF5
                try:
                    if c.suffix == ".keras" and (c.with_suffix(".h5")).exists():
                        alt = c.with_suffix(".h5")
                        mdl = tf.keras.models.load_model(str(alt))
                        print(f"[KerasASV] Loaded model from (fallback .h5): {alt}")
                        return mdl
                except Exception:
                    pass
                continue

        # dacă am ajuns aici, toate încercările au eșuat
        raise RuntimeError(
            "Eșec la încărcarea modelului Keras. Verifică formatul fișierului:\n"
            " - Dacă e HDF5, păstrează extensia .h5\n"
            " - Dacă e noul format Keras 3, extensia trebuie .keras (zip)\n"
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

    # ---------- preprocessing (legacy / non-strict) ----------
    @staticmethod
    def _interp_1d(vec: np.ndarray, out_dim: int) -> np.ndarray:
        D = vec.shape[0]
        if D == out_dim:
            return vec
        old = np.linspace(0.0, 1.0, num=D, dtype=np.float32)
        new = np.linspace(0.0, 1.0, num=out_dim, dtype=np.float32)
        return np.interp(new, old, vec).astype(np.float32)

    def _resize_features(self, X: np.ndarray) -> np.ndarray:
        B, Din = X.shape
        if Din == self.input_dim:
            return X.astype(np.float32, copy=False)
        out = np.zeros((B, self.input_dim), dtype=np.float32)
        for i in range(B):
            out[i] = self._interp_1d(X[i], self.input_dim)
        return out

    def _prepare(self, feats_np: np.ndarray) -> np.ndarray:
        if self.strict:
            if feats_np.shape[1] != self.input_dim:
                raise ValueError(f"[ASVmodel] STRICT mismatch: X={feats_np.shape[1]} vs model={self.input_dim}")
            return feats_np.astype(np.float32, copy=False)
        X = self._resize_features(feats_np)
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                pass
        return X

    # ---------- inference ----------
    def _post(self, preds) -> np.ndarray:
        p = np.array(preds)
        if p.ndim == 2:
            ex = np.exp(p - p.max(axis=1, keepdims=True))
            prob = ex / ex.sum(axis=1, keepdims=True)
            idx = min(self.bona_index, prob.shape[1] - 1)
            return prob[:, idx].astype(np.float32)
        return p.reshape(-1).astype(np.float32)

    def predict_prob_bonafide(self, feats_np: np.ndarray) -> np.ndarray:
        X = self._prepare(feats_np)
        preds = self.model.predict(X, verbose=0)
        return self._post(preds)

    def predict_prob_bonafide_prepared(self, X: np.ndarray, strict_dim: bool = True, apply_scaler: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if strict_dim and X.shape[1] != self.input_dim:
            raise ValueError(f"[ASVmodel] STRICT mismatch: X={X.shape[1]} vs model={self.input_dim}")
        if apply_scaler and self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                pass
        preds = self.model.predict(X, verbose=0)
        return self._post(preds)
