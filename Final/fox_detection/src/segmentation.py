"""
segmentation.py
Segment long audio recordings into fixed-length or energy-based clips,
process entire directories, and build a consolidated manifest CSV.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from src.audio_utils import load_audio, normalise_audio


# ── Supported audio extensions ────────────────────────────────────────────────

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _is_audio_file(path: str) -> bool:
    """Return True if *path* has a recognised audio extension."""
    return Path(path).suffix.lower() in _AUDIO_EXTS


# ── 1. Fixed-length segmentation ─────────────────────────────────────────────

def segment_fixed(
    y: np.ndarray,
    sr: int,
    clip_duration: float = 3.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Slice a waveform into fixed-length overlapping clips.

    The last clip is zero-padded if it is shorter than *clip_duration*.

    Parameters
    ----------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    clip_duration : float, optional
        Length of each clip in seconds (default 3.0).
    overlap : float, optional
        Overlap between consecutive clips in seconds (default 0.5).

    Returns
    -------
    list[np.ndarray]
        List of 1-D waveform clips, each of length ``int(sr * clip_duration)``.
    """
    clip_samples = int(sr * clip_duration)
    hop_samples = int(sr * (clip_duration - overlap))
    if hop_samples <= 0:
        raise ValueError("overlap must be smaller than clip_duration")

    clips: list[np.ndarray] = []
    start = 0
    while start < len(y):
        end = start + clip_samples
        clip = y[start:end]
        # Zero-pad the last clip if necessary
        if len(clip) < clip_samples:
            clip = np.pad(clip, (0, clip_samples - len(clip)), mode="constant")
        clips.append(clip)
        start += hop_samples
    return clips


# ── 2. Energy-based segmentation ─────────────────────────────────────────────

def segment_energy(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_threshold_db: float = -40.0,
    min_event_duration: float = 0.5,
    pad_seconds: float = 0.1,
) -> list[np.ndarray]:
    """Detect high-energy regions and return padded clips.

    Steps:
    1. Compute frame-level RMS energy.
    2. Convert to dB (relative to peak RMS).
    3. Keep frames above *energy_threshold_db*.
    4. Merge nearby active regions separated by less than *min_event_duration*.
    5. Pad each region by *pad_seconds* on both sides and extract the clip.

    Parameters
    ----------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    frame_length : int, optional
        STFT frame length (default 2048).
    hop_length : int, optional
        STFT hop length (default 512).
    energy_threshold_db : float, optional
        Threshold in dB below peak RMS to consider as "active" (default -40).
    min_event_duration : float, optional
        Minimum event duration in seconds; shorter events are dropped
        (default 0.5).
    pad_seconds : float, optional
        Padding added before and after each event in seconds (default 0.1).

    Returns
    -------
    list[np.ndarray]
        List of variable-length waveform clips for each detected event.
    """
    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))

    # Boolean mask of active frames
    active = rms_db >= energy_threshold_db

    # Convert frame indices → sample indices
    def _frame_to_sample(frame_idx: int) -> int:
        return int(frame_idx * hop_length)

    # Find contiguous active regions (start_frame, end_frame)
    regions: list[tuple[int, int]] = []
    in_region = False
    start_frame = 0
    for i, a in enumerate(active):
        if a and not in_region:
            start_frame = i
            in_region = True
        elif not a and in_region:
            regions.append((start_frame, i))
            in_region = False
    if in_region:
        regions.append((start_frame, len(active)))

    if not regions:
        return []

    # Merge nearby regions
    min_gap_frames = int(min_event_duration * sr / hop_length)
    merged: list[tuple[int, int]] = [regions[0]]
    for start_f, end_f in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start_f - prev_end < min_gap_frames:
            merged[-1] = (prev_start, end_f)
        else:
            merged.append((start_f, end_f))

    # Drop short events and extract padded clips
    min_event_samples = int(min_event_duration * sr)
    pad_samples = int(pad_seconds * sr)
    clips: list[np.ndarray] = []
    for start_f, end_f in merged:
        s = max(0, _frame_to_sample(start_f) - pad_samples)
        e = min(len(y), _frame_to_sample(end_f) + pad_samples)
        if (e - s) >= min_event_samples:
            clips.append(y[s:e])

    return clips


# ── 3. Process a directory ────────────────────────────────────────────────────

def process_directory(
    input_dir: str,
    output_dir: str,
    label: str,
    sr: int = 22050,
    clip_duration: float = 3.0,
    overlap: float = 0.5,
    method: str = "fixed",
) -> pd.DataFrame:
    """Load, normalise, segment, and save clips for every audio file in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing raw audio files.
    output_dir : str
        Root output directory; clips are saved under ``output_dir/<label>/``.
    label : str
        Class label (e.g. ``"fox"`` or ``"nonfox"``).
    sr : int, optional
        Target sample rate (default 22 050).
    clip_duration : float, optional
        Clip duration for fixed segmentation (default 3.0 s).
    overlap : float, optional
        Overlap for fixed segmentation (default 0.5 s).
    method : str, optional
        ``"fixed"`` or ``"energy"`` (default ``"fixed"``).

    Returns
    -------
    pd.DataFrame
        One row per clip with columns:
        ``file_id, source_file, clip_path, label, start_sec, end_sec``.
    """
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    rows: list[dict] = []
    audio_files = sorted(
        f for f in os.listdir(input_dir) if _is_audio_file(f)
    )

    for fname in audio_files:
        fpath = os.path.join(input_dir, fname)
        y, sr_out = load_audio(fpath, target_sr=sr)
        y = normalise_audio(y)

        if method == "fixed":
            clips = segment_fixed(y, sr_out, clip_duration=clip_duration, overlap=overlap)
        elif method == "energy":
            clips = segment_energy(y, sr_out)
        else:
            raise ValueError(f"Unknown segmentation method: {method!r}")

        hop_samples = int(sr_out * (clip_duration - overlap)) if method == "fixed" else None

        for i, clip in enumerate(clips):
            file_id = uuid.uuid4().hex[:12]
            clip_fname = f"{Path(fname).stem}_{i:04d}_{file_id}.wav"
            clip_path = os.path.join(label_dir, clip_fname)
            sf.write(clip_path, clip, sr_out)

            if method == "fixed" and hop_samples is not None:
                start_sec = round(i * hop_samples / sr_out, 4)
                end_sec = round(start_sec + clip_duration, 4)
            else:
                start_sec = 0.0
                end_sec = round(len(clip) / sr_out, 4)

            rows.append(
                {
                    "file_id": file_id,
                    "source_file": fname,
                    "clip_path": os.path.join(
                        os.path.basename(output_dir), label, clip_fname
                    ),
                    "label": label,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )

    return pd.DataFrame(rows)


# ── 4. Build manifest ────────────────────────────────────────────────────────

def build_manifest(
    fox_dir: str,
    nonfox_dir: str,
    out_dir: str = "data/clips",
    out_csv: str = "data/manifest.csv",
    sr: int = 22050,
    clip_duration: float = 3.0,
    overlap: float = 0.5,
    method: str = "fixed",
) -> pd.DataFrame:
    """Segment both classes and save a combined manifest CSV.

    Parameters
    ----------
    fox_dir : str
        Directory with raw fox audio files.
    nonfox_dir : str
        Directory with raw non-fox audio files.
    out_dir : str, optional
        Root clip output directory (default ``"data/clips"``).
    out_csv : str, optional
        Path for the combined manifest CSV (default ``"data/manifest.csv"``).
    sr : int, optional
        Target sample rate (default 22 050).
    clip_duration : float, optional
        Clip duration in seconds (default 3.0).
    overlap : float, optional
        Overlap in seconds (default 0.5).
    method : str, optional
        ``"fixed"`` or ``"energy"`` (default ``"fixed"``).

    Returns
    -------
    pd.DataFrame
        Combined manifest with columns:
        ``file_id, source_file, clip_path, label, start_sec, end_sec``.
    """
    print(f"Processing fox audio from: {fox_dir}")
    df_fox = process_directory(
        fox_dir, out_dir, label="fox",
        sr=sr, clip_duration=clip_duration, overlap=overlap, method=method,
    )
    print(f"  → {len(df_fox)} fox clips")

    print(f"Processing non-fox audio from: {nonfox_dir}")
    df_nonfox = process_directory(
        nonfox_dir, out_dir, label="nonfox",
        sr=sr, clip_duration=clip_duration, overlap=overlap, method=method,
    )
    print(f"  → {len(df_nonfox)} non-fox clips")

    df = pd.concat([df_fox, df_nonfox], ignore_index=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nManifest saved to {out_csv}  ({len(df)} total clips)")
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment raw audio into fixed-length clips and build a manifest."
    )
    parser.add_argument(
        "--fox_dir", type=str, default="data/raw/fox",
        help="Directory containing raw fox audio files.",
    )
    parser.add_argument(
        "--nonfox_dir", type=str, default="data/raw/nonfox",
        help="Directory containing raw non-fox audio files.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/clips",
        help="Root output directory for segmented clips.",
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifest.csv",
        help="Path for the output manifest CSV.",
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Target sample rate (default: 22050).",
    )
    parser.add_argument(
        "--clip_duration", type=float, default=3.0,
        help="Clip duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Overlap between clips in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--method", type=str, default="fixed", choices=["fixed", "energy"],
        help="Segmentation method: 'fixed' or 'energy' (default: fixed).",
    )
    args = parser.parse_args()

    build_manifest(
        fox_dir=args.fox_dir,
        nonfox_dir=args.nonfox_dir,
        out_dir=args.out_dir,
        out_csv=args.manifest,
        sr=args.sr,
        clip_duration=args.clip_duration,
        overlap=args.overlap,
        method=args.method,
    )
