#!/bin/bash
set -e

echo "=== Step 1: Segment audio ==="
python -m src.segmentation \
  --fox_dir data/raw/fox \
  --nonfox_dir data/raw/nonfox \
  --out_dir data/clips \
  --manifest data/manifest.csv

echo "=== Step 2: Extract features ==="
python -m src.features \
  --manifest data/manifest.csv \
  --feature_dir data/features/ \
  --spec_dir data/spectrograms/ \
  --mode both

echo "=== Step 3: Train baseline ==="
python -m src.baseline_model \
  --manifest data/manifest.csv \
  --feature_dir data/features/ \
  --model_dir models/baseline/

echo "=== Step 4: Train CNN ==="
python -m src.train_cnn \
  --manifest data/manifest.csv \
  --spec_dir data/spectrograms/ \
  --model_dir models/cnn/ \
  --backbone efficientnet_b0 \
  --epochs 30

echo "=== Step 5: Evaluate both models ==="
python -m src.evaluate --model_type baseline \
  --model_path models/baseline/model.pkl \
  --manifest data/manifest.csv \
  --feature_dir data/features/

python -m src.evaluate --model_type cnn \
  --model_path models/cnn/best.pt \
  --manifest data/manifest.csv \
  --spec_dir data/spectrograms/

echo "=== Step 6: Launch demo ==="
python src/demo.py \
  --cnn_model models/cnn/best.pt \
  --baseline_model models/baseline/model.pkl
