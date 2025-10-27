
import csv, torch, soundfile as sf
from pathlib import Path
from typing import Dict, Any

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def save_wave(path: Path, wave: torch.Tensor, sr: int):
    ensure_parent(path)
    w = wave.detach().cpu().numpy()
    sf.write(str(path), w, sr)

def append_csv_row(path: Path, row: Dict[str, Any]):
    ensure_parent(path)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
