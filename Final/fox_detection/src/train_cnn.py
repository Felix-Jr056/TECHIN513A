"""
train_cnn.py
End-to-end CNN training loop with weighted loss, cosine annealing LR,
early stopping, and training-curve visualisation.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cnn_model import FoxCNN
from src.dataset import FoxSpectrogramDataset


# ── Training function ─────────────────────────────────────────────────────────

def train_cnn(
    manifest_csv: str,
    spec_dir: str,
    model_dir: str = "models/cnn/",
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 7,
    img_size: tuple[int, int] = (128, 128),
    num_workers: int = 0,
    device: str | None = None,
) -> str:
    """Train a CNN on spectrogram images and return the path to the best
    checkpoint.

    Parameters
    ----------
    manifest_csv : str
        Path to the manifest CSV.
    spec_dir : str
        Root directory for spectrogram PNGs (``fox/`` and ``nonfox/`` sub-dirs).
    model_dir : str
        Directory for saving checkpoints and training curves.
    backbone : str
        Backbone architecture name (default ``'efficientnet_b0'``).
    pretrained : bool
        Use ImageNet pre-trained weights (default ``True``).
    epochs : int
        Maximum training epochs (default 30).
    batch_size : int
        Mini-batch size (default 32).
    lr : float
        Initial learning rate for AdamW (default 1e-3).
    weight_decay : float
        AdamW weight decay (default 1e-4).
    dropout : float
        Dropout probability in the classifier head (default 0.3).
    patience : int
        Early-stopping patience in epochs (default 7).
    img_size : tuple[int, int]
        Image resize target (default ``(128, 128)``).
    num_workers : int
        DataLoader workers (default 0).
    device : str or None
        ``'cuda'``, ``'mps'``, or ``'cpu'``.  Auto-detected if ``None``.

    Returns
    -------
    str
        Path to the best saved checkpoint (``models/cnn/best.pt``).
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # ── Datasets & loaders ────────────────────────────────────────────
    train_ds = FoxSpectrogramDataset(
        manifest_csv, spec_dir, split="train",
        img_size=img_size, augment=True,
    )
    val_ds = FoxSpectrogramDataset(
        manifest_csv, spec_dir, split="val",
        img_size=img_size, augment=False,
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = FoxCNN(
        backbone=backbone, pretrained=pretrained,
        num_classes=2, dropout=dropout,
    ).to(dev)

    # ── Loss (weighted) ──────────────────────────────────────────────
    class_weights = FoxSpectrogramDataset.get_class_weights(train_ds.labels).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.tolist()}")

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_path = os.path.join(model_dir, "best.pt")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_train = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_train += images.size(0)
        train_loss = running_loss / max(n_train, 1)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        n_val = 0
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images, labels = images.to(dev), labels.to(dev)
                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * images.size(0)
                n_val += images.size(0)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
        val_loss = val_running_loss / max(n_val, 1)
        val_f1 = float(f1_score(all_labels, all_preds, zero_division=0))

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1={val_f1:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpoint & early stopping ───────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": backbone,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  ★ Best model saved (val_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete. Best epoch: {best_epoch}  "
          f"Best val F1: {best_val_f1:.4f}")

    # ── Plot training curves ──────────────────────────────────────────
    _plot_training_curves(history, model_dir)

    return best_path


# ── Plotting helper ───────────────────────────────────────────────────────────

def _plot_training_curves(history: dict[str, list[float]], model_dir: str) -> None:
    """Save loss and F1 training curves to ``model_dir/training_curves.png``."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1
    ax2.plot(epochs_range, history["val_f1"], label="Val F1", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(model_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN on spectrogram images for fox detection."
    )
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--spec_dir", type=str, default="data/spectrograms/")
    parser.add_argument("--model_dir", type=str, default="models/cnn/")
    parser.add_argument(
        "--backbone", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "mobilenet_v3_small"],
    )
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_cnn(
        manifest_csv=args.manifest,
        spec_dir=args.spec_dir,
        model_dir=args.model_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers,
        device=args.device,
    )
