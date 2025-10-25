from pathlib import Path

def load_keras_model(model_path: Path):
    print("[ASV] loading keras model…")
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    try:
        import keras
        return keras.models.load_model(str(model_path), compile=False)
    except Exception:
        import tensorflow as tf
        return tf.keras.models.load_model(str(model_path), compile=False)
