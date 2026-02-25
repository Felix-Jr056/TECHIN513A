"""
dataset.py
PyTorch Dataset for loading fox / non-fox spectrogram images with
stratified train/val/test splitting and optional data augmentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms


# ── ImageNet normalisation constants ──────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Custom augmentation transforms ───────────────────────────────────────────

class FrequencyMask:
    """Zero out up to *max_mask* consecutive frequency (row) bands."""

    def __init__(self, max_mask: int = 20) -> None:
        self.max_mask = max_mask

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor: (C, H, W)"""
        _, h, _ = tensor.shape
        mask_len = torch.randint(0, min(self.max_mask, h) + 1, (1,)).item()
        if mask_len == 0:
            return tensor
        start = torch.randint(0, h - mask_len + 1, (1,)).item()
        tensor[:, start : start + mask_len, :] = 0.0
        return tensor


class TimeMask:
    """Zero out up to *max_mask* consecutive time (column) frames."""

    def __init__(self, max_mask: int = 20) -> None:
        self.max_mask = max_mask

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor: (C, H, W)"""
        _, _, w = tensor.shape
        mask_len = torch.randint(0, min(self.max_mask, w) + 1, (1,)).item()
        if mask_len == 0:
            return tensor
        start = torch.randint(0, w - mask_len + 1, (1,)).item()
        tensor[:, :, start : start + mask_len] = 0.0
        return tensor


class AdditiveGaussianNoise:
    """Add Gaussian noise with std *sigma*."""

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.sigma * torch.randn_like(tensor)


# ── Dataset class ─────────────────────────────────────────────────────────────

class FoxSpectrogramDataset(Dataset):
    """PyTorch Dataset that loads pre-generated spectrogram PNGs for the
    fox / non-fox binary classification task.

    Parameters
    ----------
    manifest_csv : str
        Path to the manifest CSV (columns: ``file_id``, ``clip_path``,
        ``label``, …).
    spec_dir : str
        Root directory containing ``fox/`` and ``nonfox/`` spectrogram PNGs.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    val_frac : float
        Fraction of data for validation (default 0.15).
    test_frac : float
        Fraction of data for test (default 0.15).
    img_size : tuple[int, int]
        Target image size ``(H, W)`` (default ``(128, 128)``).
    augment : bool
        If ``True`` (and ``split == 'train'``), apply data augmentation:
        random horizontal flip, frequency masking, time masking, and
        additive Gaussian noise.
    random_state : int
        Random seed for the stratified split (default 42).
    """

    LABEL_MAP = {"fox": 1, "nonfox": 0}

    def __init__(
        self,
        manifest_csv: str,
        spec_dir: str,
        split: str = "train",
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        img_size: tuple[int, int] = (128, 128),
        augment: bool = False,
        random_state: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"

        df = pd.read_csv(manifest_csv)
        df["int_label"] = df["label"].map(self.LABEL_MAP)

        # ── Stratified split: train / temp → val / test ──────────────
        train_idx, temp_idx = train_test_split(
            df.index,
            test_size=val_frac + test_frac,
            stratify=df["int_label"],
            random_state=random_state,
        )
        relative_test = test_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            stratify=df.loc[temp_idx, "int_label"],
            random_state=random_state,
        )

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        subset = df.loc[split_map[split]].reset_index(drop=True)

        self.file_ids: list[str] = subset["file_id"].astype(str).tolist()
        self.labels: list[int] = subset["int_label"].tolist()
        self.label_strs: list[str] = subset["label"].tolist()
        self.spec_dir = spec_dir
        self.img_size = img_size
        self.split = split

        # ── Build transforms ─────────────────────────────────────────
        base_transforms: list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),             # → [0, 1], shape (C, H, W)
        ]

        if augment and split == "train":
            post_tensor: list = [
                transforms.RandomHorizontalFlip(p=0.5),
                FrequencyMask(max_mask=20),
                TimeMask(max_mask=20),
                AdditiveGaussianNoise(sigma=0.01),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        else:
            post_tensor = [
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

        self.transform = transforms.Compose(base_transforms + post_tensor)

    # ── Required overrides ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return ``(image_tensor, label)`` where image_tensor has shape
        ``(3, H, W)`` and label is 0 or 1.
        """
        file_id = self.file_ids[idx]
        label_str = self.label_strs[idx]
        label = self.labels[idx]

        png_path = os.path.join(self.spec_dir, label_str, f"{file_id}.png")
        img = Image.open(png_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label

    # ── Utility ───────────────────────────────────────────────────────

    @staticmethod
    def get_class_weights(labels: list[int] | np.ndarray) -> torch.Tensor:
        """Compute inverse-frequency class weights for a weighted loss.

        Parameters
        ----------
        labels : array-like of int
            Label array (0s and 1s).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_classes,)`` with weight per class.
        """
        labels_arr = np.asarray(labels)
        classes = np.unique(labels_arr)
        n_samples = len(labels_arr)
        weights = []
        for c in sorted(classes):
            count = (labels_arr == c).sum()
            weights.append(n_samples / (len(classes) * count))
        return torch.tensor(weights, dtype=torch.float32)
