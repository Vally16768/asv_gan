#!/usr/bin/env bash
set -euo pipefail

# Unde vrei să stea datele (poți schimba)
DATA_DIR="${1:-database/data/asvspoof2019}"

# Link direct spre LA.zip de pe Edinburgh DataShare (poate avea alt "sequence" în timp)
LA_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?isAllowed=y&sequence=3"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[*] Destination: $(pwd)"

if [ ! -f LA.zip ]; then
  echo "[*] Downloading ASVspoof2019 LA.zip (resume enabled)..."
  # încearcă curl, apoi wget ca fallback
  (curl -L -C - -o LA.zip "$LA_URL") || wget --content-disposition -c "$LA_URL" -O LA.zip
fi

echo "[*] Extracting LA.zip..."
unzip -q -o LA.zip

# LA.zip se dezarhivează într-un folder 'LA/' care conține:
#  ASVspoof2019_LA_train/, _dev/, _eval/, și ASVspoof2019_LA_cm_protocols/
if [ -d LA ]; then
  shopt -s dotglob
  mv LA/* . || true
  rmdir LA || true
fi

echo "[*] Done. Found:"
ls -1d ASVspoof2019_LA_* ASVspoof2019_LA_cm_protocols | sed 's/^/   - /'

echo "[i] Codec: FLAC. License: ODC-By (atribution)."
