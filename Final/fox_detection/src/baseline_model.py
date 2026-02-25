"""
baseline_model.py
Train and evaluate scikit-learn baseline classifiers (SVM, Random Forest,
Gradient Boosting) on MFCC feature vectors.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── Classifier wrapper ────────────────────────────────────────────────────────

class BaselineClassifier:
    """Scikit-learn pipeline wrapper supporting SVM, Random Forest, and
    Gradient Boosting classifiers.

    Parameters
    ----------
    model_type : str
        One of ``'svm'``, ``'random_forest'``, or ``'gradient_boosting'``.
    """

    _MODEL_MAP: dict[str, Any] = {
        "svm": lambda: SVC(kernel="rbf", probability=True, random_state=42),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
    }

    def __init__(self, model_type: str = "svm") -> None:
        if model_type not in self._MODEL_MAP:
            raise ValueError(
                f"Unknown model_type {model_type!r}. "
                f"Choose from {list(self._MODEL_MAP)}"
            )
        self.model_type = model_type
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", self._MODEL_MAP[model_type]()),
            ]
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the pipeline on training data and print training accuracy.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix ``(n_samples, n_features)``.
        y_train : np.ndarray
            Label array ``(n_samples,)``.
        """
        self.pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.pipeline.predict(X_train))
        print(f"[{self.model_type}] Training accuracy: {train_acc:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_cm_path: str | None = None,
    ) -> dict:
        """Evaluate the model on a held-out set.

        Parameters
        ----------
        X_test : np.ndarray
            Feature matrix.
        y_test : np.ndarray
            True labels.
        save_cm_path : str or None, optional
            If given, save the confusion-matrix heatmap to this path.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``classification_report``.
        """
        y_pred = self.pipeline.predict(X_test)

        labels = [0, 1]
        target_names = ["nonfox", "fox"]

        metrics: dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred,
                labels=labels, target_names=target_names, zero_division=0,
            ),
        }

        # Confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["nonfox", "fox"],
            yticklabels=["nonfox", "fox"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {self.model_type}")
        fig.tight_layout()

        if save_cm_path:
            os.makedirs(os.path.dirname(save_cm_path) or ".", exist_ok=True)
            fig.savefig(save_cm_path, dpi=150)
            print(f"  Confusion matrix saved to {save_cm_path}")
        plt.close(fig)

        return metrics

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = "models/baseline/model.pkl") -> None:
        """Pickle the fitted pipeline to *path*.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model_type": self.model_type, "pipeline": self.pipeline}, f
            )
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineClassifier":
        """Load a previously saved :class:`BaselineClassifier`.

        Parameters
        ----------
        path : str
            Path to the ``.pkl`` file.

        Returns
        -------
        BaselineClassifier
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model_type = data["model_type"]
        obj.pipeline = data["pipeline"]
        return obj


# ── Convenience training function ─────────────────────────────────────────────

def train_baseline(
    manifest_csv: str,
    feature_dir: str = "data/features/",
    model_dir: str = "models/baseline/",
    sr: int = 22050,
    n_mfcc: int = 40,
) -> BaselineClassifier:
    """End-to-end baseline training: load features, split, train three
    classifiers, and return the best one.

    Parameters
    ----------
    manifest_csv : str
        Path to the manifest CSV.
    feature_dir : str, optional
        Directory for cached ``.npy`` MFCC files.
    model_dir : str, optional
        Directory to save the best model and artefacts.
    sr : int, optional
        Sample rate for feature extraction (default 22 050).
    n_mfcc : int, optional
        Number of MFCC coefficients (default 40).

    Returns
    -------
    BaselineClassifier
        The best classifier (by validation F1).
    """
    from src.features import extract_mfcc_dataset  # avoid circular at module level

    # 1. Load / extract features
    print("Loading MFCC features …")
    X, y = extract_mfcc_dataset(manifest_csv, feature_dir=feature_dir,
                                sr=sr, n_mfcc=n_mfcc)

    # 2. Stratified split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 3. Train all three models, pick the best by val F1
    model_types = ["svm", "random_forest", "gradient_boosting"]
    best_f1 = -1.0
    best_clf: BaselineClassifier | None = None

    for mt in model_types:
        print(f"\n── {mt} ──")
        clf = BaselineClassifier(model_type=mt)
        clf.train(X_train, y_train)
        val_metrics = clf.evaluate(
            X_val, y_val,
            save_cm_path=os.path.join(model_dir, f"cm_val_{mt}.png"),
        )
        print(f"  Val → acc={val_metrics['accuracy']:.4f}  "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_clf = clf

    assert best_clf is not None
    print(f"\n★ Best model: {best_clf.model_type} (val F1={best_f1:.4f})")

    # 4. Evaluate best on test set
    print("\n── Test-set evaluation ──")
    test_metrics = best_clf.evaluate(
        X_test, y_test,
        save_cm_path=os.path.join(model_dir, "confusion_matrix.png"),
    )
    print(test_metrics["classification_report"])

    # 5. Save
    best_clf.save(os.path.join(model_dir, "model.pkl"))
    return best_clf


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline classifiers on MFCC features."
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifest.csv",
        help="Path to the manifest CSV.",
    )
    parser.add_argument(
        "--feature_dir", type=str, default="data/features/",
        help="Directory for cached .npy MFCC files.",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models/baseline/",
        help="Output directory for the saved model and artefacts.",
    )
    args = parser.parse_args()

    train_baseline(
        manifest_csv=args.manifest,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
    )
