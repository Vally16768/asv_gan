from pathlib import Path
import numpy as np, pandas as pd
from asvspoof.features import extract_features_for_path
from asvspoof.config import ExtractConfig
from asvspoof.combos import group_columns_from_df, columns_for_code, normalize_codes_to_sorted_unique

def _try_load_scaler(path: Path):
    try:
        import joblib
        return joblib.load(path) if path and path.exists() else None
    except Exception:
        return None

def build_keras_input_vector(audio_path: Path, combo: str, sr: int, scaler_path: Path|None):
    cfg = ExtractConfig(sampling_rate=sr)
    feats = extract_features_for_path(audio_path, cfg)
    row = {"split":"infer","file_id":audio_path.stem,"path":str(audio_path),"label":None,"target":None}
    row.update(feats)
    df = pd.DataFrame([row])

    groups = group_columns_from_df(df)
    combo_norm = normalize_codes_to_sorted_unique([combo])[0]
    cols = columns_for_code(combo_norm, groups)

    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    scaler = _try_load_scaler(scaler_path) if scaler_path else None
    if scaler is not None:
        X = scaler.transform(X)
    return X  # [1, D]
