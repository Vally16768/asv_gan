# asvspoof/combos.py — combos pe litere, cu completare pentru grupuri lipsă + aliases de prefix
from __future__ import annotations
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

from .config import (
    FEATURES_LIST,
    FEATURE_NAME_MAPPING,
    FEATURE_NAME_REVERSE_MAPPING,
    GROUP_ALIASES,
)

META_COLS = {"split", "file_id", "path", "label", "target"}
_DEFAULT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _norm_group(name: str) -> str:
    return str(name).strip().lower()


def _effective_letter_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Returnează (forward, reverse) complete:
      forward: group -> letter  (ex: 'mfcc' -> 'A')
      reverse: letter -> group  (ex: 'A' -> 'mfcc')

    - Folosește FEATURE_NAME_REVERSE_MAPPING dacă există, dar **completează**
      automat grupurile lipsă din FEATURES_LIST cu litere nefolosite.
    - Toate numele de grup sunt normalizate la lowercase.
    """
    feats = [_norm_group(g) for g in FEATURES_LIST]
    if len(feats) > len(_DEFAULT_LETTERS):
        raise ValueError(
            f"Prea multe grupuri ({len(feats)}) pentru maparea A..Z. "
            f"Reduceți grupurile sau extindeți alfabetul."
        )

    forward: Dict[str, str] = {}
    used_letters: Set[str] = set()

    if isinstance(FEATURE_NAME_REVERSE_MAPPING, dict) and FEATURE_NAME_REVERSE_MAPPING:
        for raw_letter, raw_group in FEATURE_NAME_REVERSE_MAPPING.items():
            L = str(raw_letter).strip().upper()
            if len(L) != 1 or L not in _DEFAULT_LETTERS:
                continue
            g = _norm_group(raw_group)
            forward[g] = L
            used_letters.add(L)

    available_letters = [L for L in _DEFAULT_LETTERS if L not in used_letters]
    missing = [g for g in feats if g not in forward]
    if len(missing) > len(available_letters):
        raise ValueError("Nu mai sunt litere disponibile pentru a mapa toate grupurile lipsă.")
    for g, L in zip(missing, available_letters):
        forward[g] = L
        used_letters.add(L)

    reverse: Dict[str, str] = {L: g for g, L in forward.items()}
    return forward, reverse


def group_columns_from_df(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Alocă fiecare coloană de feature în grupul corespunzător pe baza prefixelor
    definite în GROUP_ALIASES. Acceptă mai multe prefixe per grup.
    """
    feats = [_norm_group(g) for g in FEATURES_LIST]
    aliases = {
        g: [g] + [a.strip().lower() for a in GROUP_ALIASES.get(g, [])]
        for g in feats
    }
    groups: Dict[str, List[str]] = {g: [] for g in feats}

    for col in df.columns:
        if col in META_COLS:
            continue
        col_l = col.strip().lower()
        for g in feats:
            for pref in aliases[g]:
                if col_l == pref or col_l.startswith(pref + "_"):
                    groups[g].append(col)
                    break
            else:
                continue
            break

    for g in groups:
        groups[g] = sorted(groups[g])
    return groups


def all_combo_codes() -> List[str]:
    forward, _ = _effective_letter_maps()
    letters = [forward[_norm_group(g)] for g in FEATURES_LIST]
    codes: List[str] = []
    for r in range(1, len(letters) + 1):
        for combo in itertools.combinations(letters, r):
            codes.append("".join(combo))
    return codes


def normalize_codes_to_sorted_unique(codes: Iterable[str]) -> List[str]:
    _, reverse = _effective_letter_maps()
    allowed: Set[str] = set(reverse.keys())
    normed: Set[str] = set()
    for raw in codes:
        s = "".join(sorted(ch.upper() for ch in str(raw) if ch.strip()))
        if s and set(s).issubset(allowed):
            normed.add(s)
    return sorted(normed)


def columns_for_code(code: str, group_cols: Dict[str, List[str]]) -> List[str]:
    _, reverse = _effective_letter_maps()
    cols: List[str] = []
    for ch in code:
        if ch not in reverse:
            raise KeyError(f"Unknown feature letter '{ch}' in code '{code}'")
        g = reverse[ch]
        cols.extend(group_cols.get(g, []))
    return cols


def write_npz(out_path: Path, X: np.ndarray, y: Optional[np.ndarray], columns: List[str], combo_code: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32, copy=False),
        y=(y if y is not None else np.array([], dtype=np.int16)),
        columns=np.array(columns),
        combo_code=np.array(combo_code),
    )


def materialize_combos(feat_df: pd.DataFrame, out_dir: Path, codes: List[str]) -> None:
    out_dir = Path(out_dir)
    meta_path = out_dir / "combos_meta.json"

    forward, reverse = _effective_letter_maps()
    group_cols = group_columns_from_df(feat_df)

    meta = {
        "features_list": [_norm_group(g) for g in FEATURES_LIST],
        "feature_name_mapping": FEATURE_NAME_MAPPING,
        "feature_letter_map": forward,                 # group -> letter
        "feature_letter_reverse_map": reverse,         # letter -> group
        "groups_to_columns": group_cols,
        "num_rows": int(len(feat_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # -------- FIX: Avoid split mixing by using masks / reset_index --------
    df_lab = feat_df[feat_df["split"].isin(["train", "val", "test"])].copy()
    df_lab.reset_index(drop=True, inplace=True)

    base_cols = sorted([c for c in df_lab.columns if c not in META_COLS])
    M = df_lab[base_cols]
    y_all = df_lab["target"].astype("int16").to_numpy()
    col_to_pos = {c: i for i, c in enumerate(base_cols)}

    split_mask = {s: (df_lab["split"] == s).to_numpy() for s in ["train", "val", "test"]}

    def slice_for(code: str) -> tuple[List[str], List[int]]:
        cols = columns_for_code(code, group_cols)
        pos = [col_to_pos[c] for c in cols]
        return cols, pos

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x  # type: ignore

    codes = normalize_codes_to_sorted_unique(codes)
    if not codes:
        raise SystemExit("No valid combo codes after normalization (check your letters).")

    for code in tqdm(codes, desc="Combos"):
        cols, pos = slice_for(code)
        if not cols:
            continue
        Ms = M.iloc[:, pos]
        for split in ["train", "val", "test"]:
            mask = split_mask.get(split)
            if mask is None or not mask.any():
                continue
            X = Ms[mask].to_numpy(dtype=np.float32, copy=False)
            y = y_all[mask]
            out_path = out_dir / "combos" / split / f"{code}.npz"
            write_npz(out_path, X, y, cols, code)
