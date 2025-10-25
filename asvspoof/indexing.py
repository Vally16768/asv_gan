# === indexing.py (load *existing* indices; no new split logic) ===
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

LABEL_MAP = {"bonafide": 1, "spoof": 0}


def _load_labeled_csv(csv_path: Path, split: str, data_root: Path) -> pd.DataFrame:
    """CSV format: path,label  (path is RELATIVE to data_root)."""
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must have columns: path,label")
    df["split"] = split
    df["abs_path"] = df["path"].map(lambda p: str((data_root / p).resolve()))
    df["target"] = df["label"].str.lower().map(LABEL_MAP)
    df["file_id"] = df["abs_path"].map(lambda p: Path(p).stem)
    return df[["split", "file_id", "abs_path", "path", "label", "target"]]


def _load_eval_list(list_path: Path, data_root: Path) -> pd.DataFrame:
    """Plain list of relative paths (unlabeled)."""
    rows: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for ln in map(str.strip, f):
            if ln:
                rows.append(ln)
    df = pd.DataFrame({"path": rows})
    df["split"] = "eval"
    df["abs_path"] = df["path"].map(lambda p: str((data_root / p).resolve()))
    df["label"] = None
    df["target"] = None
    df["file_id"] = df["abs_path"].map(lambda p: Path(p).stem)
    return df[["split", "file_id", "abs_path", "path", "label", "target"]]


def load_existing_indices(data_root: Path, index_dirname: str) -> pd.DataFrame:
    idx = data_root / index_dirname
    paths = {
        "train": idx / "train.csv",
        "val":   idx / "val.csv",
        "test":  idx / "test.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists() and k != "eval"]
    if missing:
        raise SystemExit(
            "[!] Missing required index files in " + str(idx) + "\n" +
            "    Expected: train.csv, val.csv, test.csv (eval.list optional)"
        )

    dfs: List[pd.DataFrame] = []
    if paths["train"].exists(): dfs.append(_load_labeled_csv(paths["train"], "train", data_root))
    if paths["val"].exists():   dfs.append(_load_labeled_csv(paths["val"],   "val",   data_root))
    if paths["test"].exists():  dfs.append(_load_labeled_csv(paths["test"],  "test",  data_root))

    df = pd.concat(dfs, ignore_index=True)
    return df
