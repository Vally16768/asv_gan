# === io_utils.py (write master table) ===
from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_features_tables(feat_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features_all.parquet").write_bytes(feat_df.to_parquet(index=False))
    feat_df.to_csv(out_dir / "features_all.csv", index=False)
