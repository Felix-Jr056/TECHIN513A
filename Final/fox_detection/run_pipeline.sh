#!/bin/bash
set -e

# ── Resolve Python: prefer venv, fall back to python3 ─────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "${SCRIPT_DIR}/../.venv/bin/python" ]; then
  PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON=python3
else
  echo "ERROR: No python3 or .venv/bin/python found" >&2; exit 1
fi
echo "Using Python: ${PYTHON}"

# ── Require XC_KEY for data download ──────────────────────────────────
if [ -z "${XC_KEY}" ]; then
  echo "⚠  Set XC_KEY env var to download data from Xeno-canto, e.g.:"
  echo "     export XC_KEY=your_api_key"
  echo "   Skipping download step."
else
  echo "=== Step 0: Download data from Xeno-canto ==="
  $PYTHON -m src.download_data --xc_key "${XC_KEY}"
fi

echo "=== Step 1: Segment audio ==="
$PYTHON -m src.segmentation \
  --fox_dir data/raw/fox \
  --nonfox_dir data/raw/nonfox \
  --out_dir data/clips \
  --manifest data/manifest.csv

echo "=== Step 2: Extract features ==="
$PYTHON -m src.features \
  --manifest data/manifest.csv \
  --feature_dir data/features/ \
  --spec_dir data/spectrograms/ \
  --mode both

echo "=== Step 3: Train baseline ==="
$PYTHON -m src.baseline_model \
  --manifest data/manifest.csv \
  --feature_dir data/features/ \
  --model_dir models/baseline/

echo "=== Step 4: Train CNN ==="
$PYTHON -m src.train_cnn \
  --manifest data/manifest.csv \
  --spec_dir data/spectrograms/ \
  --model_dir models/cnn/ \
  --backbone efficientnet_b0 \
  --epochs 30

echo "=== Step 5: Evaluate both models ==="
$PYTHON -m src.evaluate --model_type baseline \
  --model_path models/baseline/model.pkl \
  --manifest data/manifest.csv \
  --feature_dir data/features/

$PYTHON -m src.evaluate --model_type cnn \
  --model_path models/cnn/best.pt \
  --manifest data/manifest.csv \
  --spec_dir data/spectrograms/

echo "=== Step 6: Launch demo ==="
$PYTHON src/demo.py \
  --cnn_model models/cnn/best.pt \
  --baseline_model models/baseline/model.pkl
