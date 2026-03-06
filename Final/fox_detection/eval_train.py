"""
eval_train.py
Evaluate both baseline SVM and CNN on the TRAINING split 
to compare train vs test performance (overfitting check).
"""
import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_model import BaselineClassifier
from src.features import extract_mfcc_dataset
from src.cnn_model import FoxCNN
from src.dataset import FoxSpectrogramDataset

MANIFEST = "data/manifest.csv"
FEATURE_DIR = "data/features/"
SPEC_DIR = "data/spectrograms/"
BASELINE_PATH = "models/baseline/model.pkl"
CNN_PATH = "models/cnn/best.pt"

def eval_baseline_train():
    print("=" * 60)
    print("  BASELINE (SVM) — TRAINING SET EVALUATION")
    print("=" * 60)
    
    clf = BaselineClassifier.load(BASELINE_PATH)
    X, y = extract_mfcc_dataset(MANIFEST, feature_dir=FEATURE_DIR)
    
    # Reproduce the 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # Predict on TRAIN
    y_train_pred = clf.pipeline.predict(X_train)
    # Predict on TEST
    y_test_pred = clf.pipeline.predict(X_test)
    
    print(f"\n  Train samples: {len(y_train)}")
    print(f"  Test samples:  {len(y_test)}")
    
    print(f"\n  --- TRAIN ---")
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred, average='macro'):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_train_pred, average='macro'):.4f}")
    print(f"  F1 (macro):{f1_score(y_train, y_train_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_train, y_train_pred, target_names=['nonfox','fox'])}")
    print(f"  Confusion Matrix (train):")
    print(confusion_matrix(y_train, y_train_pred))
    
    print(f"\n  --- TEST ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred, average='macro'):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred, average='macro'):.4f}")
    print(f"  F1 (macro):{f1_score(y_test, y_test_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_test_pred, target_names=['nonfox','fox'])}")
    print(f"  Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))


def eval_cnn_train():
    print("\n" + "=" * 60)
    print("  CNN (EfficientNet-B0) — TRAINING SET EVALUATION")
    print("=" * 60)
    
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    
    ckpt = torch.load(CNN_PATH, map_location=dev, weights_only=False)
    backbone = ckpt.get("backbone", "efficientnet_b0")
    model = FoxCNN(backbone=backbone, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()
    
    # Evaluate on TRAIN split
    train_ds = FoxSpectrogramDataset(
        MANIFEST, SPEC_DIR, split="train", img_size=(128, 128), augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    
    # Evaluate on TEST split
    test_ds = FoxSpectrogramDataset(
        MANIFEST, SPEC_DIR, split="test", img_size=(128, 128), augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    for split_name, loader in [("TRAIN", train_loader), ("TEST", test_loader)]:
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(dev)
                logits = model(images)
                preds = logits.argmax(dim=1)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.cpu().tolist())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        n = len(y_true)
        print(f"\n  --- {split_name} (n={n}) ---")
        print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"  Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"  F1 (macro):{f1_score(y_true, y_pred, average='macro'):.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=['nonfox','fox'])}")
        print(f"  Confusion Matrix ({split_name.lower()}):")
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    eval_baseline_train()
    eval_cnn_train()
