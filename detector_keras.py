# detector_keras.py
from pathlib import Path

def _keras_disable_gpu_and_threads():
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

def load_keras_model(model_path: Path):
    """
    Robust loader for Keras 3:
    - Accepts new `.keras` (zip) OR legacy `.h5`
    - Tries safe_mode=False for broader compatibility
    - Falls back between keras and tf.keras
    """
    _keras_disable_gpu_and_threads()

    path = Path(model_path)
    if not path.exists():
        # try sibling .h5 / .keras
        alt = path.with_suffix(".h5") if path.suffix != ".h5" else path.with_suffix(".keras")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"[ASV] Model file not found: {model_path} (also tried {alt})")

    # 1) Try keras >=3 API first
    try:
        import keras
        try:
            return keras.models.load_model(str(path), compile=False, safe_mode=False)
        except TypeError:
            # Older keras without safe_mode kw
            return keras.models.load_model(str(path), compile=False)
    except Exception as e1:
        err1 = e1

    # 2) Fallback to tf.keras
    try:
        import tensorflow as tf
        try:
            return tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
        except TypeError:
            return tf.keras.models.load_model(str(path), compile=False)
    except Exception as e2:
        # Explain likely mismatch of extension vs format
        raise RuntimeError(
            f"[ASV] Failed to load Keras model from {path}.\n"
            f"- keras.load_model error: {type(err1).__name__}: {err1}\n"
            f"- tf.keras.load_model error: {type(e2).__name__}: {e2}\n"
            f"Tips:\n"
            f"  • If your file is legacy HDF5, ensure it has '.h5' extension (and install 'h5py').\n"
            f"  • If it’s a new Keras v3 archive, use '.keras'.\n"
            f"  • You currently have: suffix={path.suffix}"
        )
