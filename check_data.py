# check_data.py
from pathlib import Path
from utils import scan_data_folder

if __name__ == "__main__":
    root = "database/data"
    rows = scan_data_folder(root)
    if not rows:
        print(f"[warn] No audio found under {root}")
    else:
        total = len(rows)
        secs = sum(r[2] for r in rows)
        print(f"[ok] Found {total} audio files; total duration ~ {secs/3600:.2f} h")
        # afișează primele 5
        for r in rows[:5]:
            print(f" - {Path(r[0]).name}: {r[2]:.2f}s @ {r[1]}Hz")
