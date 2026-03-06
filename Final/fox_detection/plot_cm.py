"""
Generate confusion matrix plots for TRAIN and TEST sets side by side.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.baseline_model import BaselineClassifier
from src.features import extract_mfcc_dataset
from src.cnn_model import FoxCNN
from src.dataset import FoxSpectrogramDataset

MANIFEST = "data/manifest.csv"
OUT_DIR = "models/eval_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_cm_pair(cm_train, cm_test, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, label in [(axes[0], cm_train, "Train"), (axes[1], cm_test, "Test")]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["nonfox", "fox"],
                    yticklabels=["nonfox", "fox"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{label} Set")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")

# ── Baseline SVM ──
print("Baseline SVM...")
clf = BaselineClassifier.load("models/baseline/model.pkl")
X, y = extract_mfcc_dataset(MANIFEST, feature_dir="data/features/")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

cm_train_bl = confusion_matrix(y_train, clf.pipeline.predict(X_train), labels=[0,1])
cm_test_bl = confusion_matrix(y_test, clf.pipeline.predict(X_test), labels=[0,1])
plot_cm_pair(cm_train_bl, cm_test_bl, "SVM Baseline — Confusion Matrix", "cm_baseline_train_test.png")

# ── CNN ──
print("CNN (EfficientNet-B0)...")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ckpt = torch.load("models/cnn/best.pt", map_location=dev, weights_only=False)
model = FoxCNN(backbone=ckpt.get("backbone", "efficientnet_b0"), pretrained=False, num_classes=2)
model.load_state_dict(ckpt["model_state_dict"])
model.to(dev).eval()

def get_preds(split):
    ds = FoxSpectrogramDataset(MANIFEST, "data/spectrograms/", split=split, img_size=(128,128), augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    labels_all, preds_all = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            preds = model(imgs.to(dev)).argmax(1).cpu().tolist()
            labels_all.extend(labs.tolist())
            preds_all.extend(preds)
    return np.array(labels_all), np.array(preds_all)

y_train_true, y_train_pred = get_preds("train")
y_test_true, y_test_pred = get_preds("test")

cm_train_cnn = confusion_matrix(y_train_true, y_train_pred, labels=[0,1])
cm_test_cnn = confusion_matrix(y_test_true, y_test_pred, labels=[0,1])
plot_cm_pair(cm_train_cnn, cm_test_cnn, "CNN (EfficientNet-B0) — Confusion Matrix", "cm_cnn_train_test.png")

print(f"\nAll plots saved to {OUT_DIR}/")
