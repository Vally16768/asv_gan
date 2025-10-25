import os, shutil

root = "database/data/asvspoof2019"
protocols = {
    "train": f"{root}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": f"{root}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": f"{root}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

for split, proto in protocols.items():
    src = f"{root}/ASVspoof2019_LA_{split}/flac"
    dst = f"{root}/ASVspoof2019_LA_{split}/flac_bonafide"
    os.makedirs(dst, exist_ok=True)
    with open(proto) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[4] == "bonafide":
                fname = parts[1] + ".flac" if parts[1].endswith(".flac") == False else parts[1]
                src_file = os.path.join(src, parts[1] + ".flac")
                if os.path.exists(src_file):
                    shutil.copy(src_file, os.path.join(dst, os.path.basename(src_file)))
