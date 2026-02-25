"""
evaluate.py
Unified evaluation for baseline (sklearn) and CNN (PyTorch) models.
Computes metrics, plots confusion matrix / PR / ROC curves, and
provides a side-by-side model comparison table.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_baseline(model_path: str, manifest_csv: str,
                    feature_dir: str, sr: int = 22050
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run baseline inference on the test split.

    Returns (y_true, y_pred, y_prob) where y_prob is P(fox).
    """
    from src.baseline_model import BaselineClassifier
    from src.features import extract_mfcc_dataset
    from sklearn.model_selection import train_test_split

    clf = BaselineClassifier.load(model_path)
    X, y = extract_mfcc_dataset(manifest_csv, feature_dir=feature_dir, sr=sr)

    # Reproduce the same 70/15/15 split used during training
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42,
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42,
    )

    y_pred = clf.pipeline.predict(X_test)
    if hasattr(clf.pipeline, "predict_proba"):
        y_prob = clf.pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.pipeline.decision_function(X_test)

    return np.asarray(y_test), np.asarray(y_pred), np.asarray(y_prob)


def _infer_cnn(model_path: str, manifest_csv: str,
               spec_dir: str, device: str | None = None,
               img_size: tuple[int, int] = (128, 128),
               batch_size: int = 32,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CNN inference on the test split.

    Returns (y_true, y_pred, y_prob) where y_prob is P(fox).
    """
    from src.cnn_model import FoxCNN
    from src.dataset import FoxSpectrogramDataset

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    backbone = ckpt.get("backbone", "efficientnet_b0")

    model = FoxCNN(backbone=backbone, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    test_ds = FoxSpectrogramDataset(
        manifest_csv, spec_dir, split="test",
        img_size=img_size, augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(dev)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fox)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str,
) -> None:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["nonfox", "fox"],
        yticklabels=["nonfox", "fox"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str, title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str, title: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_type: str,
    model_path: str,
    manifest_csv: str,
    feature_dir: str | None = None,
    spec_dir: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate a saved model on the **test** split and produce metrics + plots.

    Parameters
    ----------
    model_type : str
        ``'baseline'`` or ``'cnn'``.
    model_path : str
        Path to the saved model (``.pkl`` for baseline, ``.pt`` for CNN).
    manifest_csv : str
        Path to the manifest CSV.
    feature_dir : str or None
        Required when ``model_type='baseline'``.
    spec_dir : str or None
        Required when ``model_type='cnn'``.
    output_dir : str or None
        Directory for saving plots.  Defaults to the parent of *model_path*.
    device : str or None
        Device for CNN inference (auto-detected if ``None``).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision_macro``, ``recall_macro``,
        ``f1_macro``, ``f1_weighted``, ``pr_auc``, ``roc_auc``,
        ``classification_report``.
    """
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────
    if model_type == "baseline":
        if feature_dir is None:
            raise ValueError("feature_dir is required for baseline evaluation")
        y_true, y_pred, y_prob = _infer_baseline(model_path, manifest_csv, feature_dir)
    elif model_type == "cnn":
        if spec_dir is None:
            raise ValueError("spec_dir is required for CNN evaluation")
        y_true, y_pred, y_prob = _infer_cnn(model_path, manifest_csv, spec_dir, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # ── Metrics ───────────────────────────────────────────────────────
    labels = [0, 1]
    target_names = ["nonfox", "fox"]

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # PR-AUC and ROC-AUC (need both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0,
    )

    # ── Print ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  Evaluation — {model_type.upper()}")
    print(f"{'═' * 50}")
    for k in ("accuracy", "precision_macro", "recall_macro",
              "f1_macro", "f1_weighted", "pr_auc", "roc_auc"):
        print(f"  {k:20s}: {metrics[k]:.4f}")
    print()
    print(metrics["classification_report"])

    # ── Plots ─────────────────────────────────────────────────────────
    tag = model_type
    _plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f"confusion_matrix_{tag}.png"),
        title=f"Confusion Matrix — {tag}",
    )
    print(f"  Saved confusion_matrix_{tag}.png")

    if len(np.unique(y_true)) > 1:
        _plot_pr_curve(
            y_true, y_prob,
            save_path=os.path.join(output_dir, f"pr_curve_{tag}.png"),
            title=f"Precision-Recall — {tag}",
        )
        print(f"  Saved pr_curve_{tag}.png")

        _plot_roc_curve(
            y_true, y_prob,
            save_path=os.path.join(output_dir, f"roc_curve_{tag}.png"),
            title=f"ROC Curve — {tag}",
        )
        print(f"  Saved roc_curve_{tag}.png")

    return metrics


# ── Model comparison ──────────────────────────────────────────────────────────

def compare_models(baseline_metrics: dict, cnn_metrics: dict) -> pd.DataFrame:
    """Print a side-by-side comparison table of two models.

    Parameters
    ----------
    baseline_metrics : dict
        Metrics dict returned by :func:`evaluate_model` for the baseline.
    cnn_metrics : dict
        Metrics dict returned by :func:`evaluate_model` for the CNN.

    Returns
    -------
    pd.DataFrame
        Comparison table (metric × model).
    """
    keys = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "pr_auc", "roc_auc",
    ]
    data = {
        "Metric": keys,
        "Baseline": [baseline_metrics.get(k, float("nan")) for k in keys],
        "CNN": [cnn_metrics.get(k, float("nan")) for k in keys],
    }
    df = pd.DataFrame(data)
    df["Δ (CNN − Baseline)"] = df["CNN"] - df["Baseline"]

    print("\n" + "═" * 60)
    print("  Model Comparison")
    print("═" * 60)
    print(df.to_string(index=False, float_format="%.4f"))
    print()
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained baseline or CNN model on the test split."
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["baseline", "cnn"],
        help="Type of model to evaluate.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the saved model file (.pkl or .pt).",
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifest.csv",
        help="Path to the manifest CSV.",
    )
    parser.add_argument(
        "--feature_dir", type=str, default=None,
        help="Directory for cached .npy MFCC files (baseline only).",
    )
    parser.add_argument(
        "--spec_dir", type=str, default=None,
        help="Directory for spectrogram PNGs (CNN only).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for saving evaluation plots (defaults to model parent dir).",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    evaluate_model(
        model_type=args.model_type,
        model_path=args.model_path,
        manifest_csv=args.manifest,
        feature_dir=args.feature_dir,
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
