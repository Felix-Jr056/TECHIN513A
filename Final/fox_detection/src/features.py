"""
features.py
Extract MFCC feature vectors and log-mel spectrogram images from audio clips
listed in a manifest CSV.  Supports caching so already-processed files are
skipped on re-runs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio_utils import (
    load_audio,
    normalise_audio,
    compute_mfcc_features,
    compute_log_mel_spectrogram,
    save_spectrogram_image,
)


# ── 1. MFCC dataset extraction ───────────────────────────────────────────────

def extract_mfcc_dataset(
    manifest_csv: str,
    feature_dir: str = "data/features/",
    sr: int = 22050,
    n_mfcc: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-clip MFCC feature vectors and cache them as ``.npy`` files.

    For each row in the manifest the function:

    1. Checks whether ``<feature_dir>/<file_id>.npy`` already exists (cache hit).
    2. If not, loads the clip, computes the MFCC feature vector via
       :func:`src.audio_utils.compute_mfcc_features`, and saves it.
    3. Collects all feature vectors into a single array.

    Parameters
    ----------
    manifest_csv : str
        Path to the manifest CSV (must contain columns
        ``file_id``, ``clip_path``, ``label``).
    feature_dir : str, optional
        Directory for cached ``.npy`` feature files (default ``"data/features/"``).
    sr : int, optional
        Target sample rate (default 22 050).
    n_mfcc : int, optional
        Number of MFCC coefficients (default 40).  The resulting feature
        vector has length ``2 * n_mfcc``.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, 2 * n_mfcc)``.
    y : np.ndarray
        Integer label array of shape ``(n_samples,)`` — 1 for *fox*, 0 for
        *nonfox*.
    """
    os.makedirs(feature_dir, exist_ok=True)
    df = pd.read_csv(manifest_csv)

    # Resolve clip paths relative to the manifest's parent directory
    manifest_root = str(Path(manifest_csv).resolve().parent)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    label_map = {"fox": 1, "nonfox": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MFCC features"):
        file_id: str = str(row["file_id"])
        npy_path = os.path.join(feature_dir, f"{file_id}.npy")

        if os.path.isfile(npy_path):
            feat = np.load(npy_path)
        else:
            clip_path = os.path.join(manifest_root, row["clip_path"])
            y_audio, _ = load_audio(clip_path, target_sr=sr)
            y_audio = normalise_audio(y_audio)
            feat = compute_mfcc_features(y_audio, sr, n_mfcc=n_mfcc)
            np.save(npy_path, feat)

        X_list.append(feat)
        y_list.append(label_map.get(row["label"], 0))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"MFCC dataset: X {X.shape}, y {y.shape}  "
          f"(fox={int((y == 1).sum())}, nonfox={int((y == 0).sum())})")
    return X, y


# ── 2. Spectrogram dataset extraction ────────────────────────────────────────

def extract_spectrogram_dataset(
    manifest_csv: str,
    spec_dir: str = "data/spectrograms/",
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmax: float = 8000.0,
    figsize: tuple[int, int] = (3, 3),
) -> None:
    """Generate and save log-mel spectrogram PNG images for every clip.

    Images are saved under ``<spec_dir>/<label>/<file_id>.png``.
    Files that already exist are skipped.

    Parameters
    ----------
    manifest_csv : str
        Path to the manifest CSV.
    spec_dir : str, optional
        Root output directory for spectrogram images
        (default ``"data/spectrograms/"``).
    sr : int, optional
        Target sample rate (default 22 050).
    n_mels : int, optional
        Number of Mel bands (default 128).
    hop_length : int, optional
        Hop length in samples (default 512).
    n_fft : int, optional
        FFT window size (default 2048).
    fmax : float, optional
        Maximum frequency for the Mel filterbank (default 8 000 Hz).
    figsize : tuple[int, int], optional
        Figure size for saved images (default ``(3, 3)``).
    """
    df = pd.read_csv(manifest_csv)
    manifest_root = str(Path(manifest_csv).resolve().parent)

    skipped = 0
    created = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Spectrograms"):
        label: str = str(row["label"])
        file_id: str = str(row["file_id"])
        label_dir = os.path.join(spec_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        png_path = os.path.join(label_dir, f"{file_id}.png")
        if os.path.isfile(png_path):
            skipped += 1
            continue

        clip_path = os.path.join(manifest_root, row["clip_path"])
        y_audio, _ = load_audio(clip_path, target_sr=sr)
        y_audio = normalise_audio(y_audio)
        spec = compute_log_mel_spectrogram(
            y_audio, sr,
            n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, fmax=fmax,
        )
        save_spectrogram_image(spec, png_path, figsize=figsize)
        created += 1

    print(f"Spectrograms: {created} created, {skipped} skipped (cached)")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MFCC features and/or spectrogram images from audio clips."
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifest.csv",
        help="Path to the manifest CSV (default: data/manifest.csv).",
    )
    parser.add_argument(
        "--feature_dir", type=str, default="data/features/",
        help="Output directory for cached .npy MFCC files (default: data/features/).",
    )
    parser.add_argument(
        "--spec_dir", type=str, default="data/spectrograms/",
        help="Output directory for spectrogram PNGs (default: data/spectrograms/).",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["mfcc", "spectrogram", "both"],
        help="Extraction mode: 'mfcc', 'spectrogram', or 'both' (default: both).",
    )
    args = parser.parse_args()

    if args.mode in ("mfcc", "both"):
        X, y = extract_mfcc_dataset(args.manifest, args.feature_dir)
        print(f"  Saved MFCC .npy files to {args.feature_dir}")

    if args.mode in ("spectrogram", "both"):
        extract_spectrogram_dataset(args.manifest, args.spec_dir)
        print(f"  Saved spectrogram PNGs to {args.spec_dir}")
