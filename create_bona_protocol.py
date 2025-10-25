# create_bona_protocol.py
import csv
from pathlib import Path
from utils import list_audio_paths

OUT = Path("protocols")
OUT.mkdir(exist_ok=True)
out_path = OUT / "la_bona_only.csv"

# Adună TOATE fișierele audio bona-fide din database/data
paths = list_audio_paths("database/data")

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "label"])  # 0 = bona-fide
    for p in sorted(paths):
        w.writerow([p, 0])

print(f"Wrote protocol CSV: {out_path}  (rows: {len(paths)})")
