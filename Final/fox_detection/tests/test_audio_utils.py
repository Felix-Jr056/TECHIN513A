"""
tests/test_audio_utils.py
Unit tests for audio_utils and segmentation helpers.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest
import soundfile as sf

# ── Ensure the project root is importable ─────────────────────────────────────
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import (
    load_audio,
    normalise_audio,
    compute_mfcc_features,
)
from src.segmentation import segment_fixed


# ── Fixtures ──────────────────────────────────────────────────────────────────

SR = 22050
DURATION = 3.0  # seconds
FREQ = 440.0    # Hz  (A4)


@pytest.fixture
def sine_wave() -> np.ndarray:
    """Return a 3-second 440 Hz sine wave at 22 050 Hz."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * FREQ * t)).astype(np.float32)


@pytest.fixture
def tmp_wav(sine_wave: np.ndarray, tmp_path: Path) -> str:
    """Write the sine wave to a temporary .wav file and return its path."""
    wav_path = str(tmp_path / "test_sine.wav")
    sf.write(wav_path, sine_wave, SR)
    return wav_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadAudio:
    """Tests for load_audio."""

    def test_returns_correct_sr(self, tmp_wav: str) -> None:
        y, sr = load_audio(tmp_wav, target_sr=SR)
        assert sr == SR

    def test_returns_float32(self, tmp_wav: str) -> None:
        y, _ = load_audio(tmp_wav, target_sr=SR)
        assert y.dtype == np.float32

    def test_mono_1d(self, tmp_wav: str) -> None:
        y, _ = load_audio(tmp_wav, target_sr=SR)
        assert y.ndim == 1

    def test_sample_count_roughly_correct(self, tmp_wav: str) -> None:
        y, _ = load_audio(tmp_wav, target_sr=SR)
        expected = int(SR * DURATION)
        # Allow ±1 sample tolerance for rounding
        assert abs(len(y) - expected) <= 1

    def test_resample(self, tmp_wav: str) -> None:
        """Loading at a different sample rate should resample."""
        new_sr = 16000
        y, sr = load_audio(tmp_wav, target_sr=new_sr)
        assert sr == new_sr
        expected = int(new_sr * DURATION)
        assert abs(len(y) - expected) <= 1


class TestNormaliseAudio:
    """Tests for normalise_audio."""

    def test_output_in_range(self, sine_wave: np.ndarray) -> None:
        normed = normalise_audio(sine_wave)
        assert np.all(normed >= -1.0)
        assert np.all(normed <= 1.0)

    def test_peak_is_one(self, sine_wave: np.ndarray) -> None:
        normed = normalise_audio(sine_wave)
        assert np.isclose(np.max(np.abs(normed)), 1.0)

    def test_silent_signal_unchanged(self) -> None:
        silent = np.zeros(SR, dtype=np.float32)
        normed = normalise_audio(silent)
        assert np.all(normed == 0.0)

    def test_preserves_dtype(self, sine_wave: np.ndarray) -> None:
        normed = normalise_audio(sine_wave)
        assert normed.dtype == np.float32


class TestComputeMfccFeatures:
    """Tests for compute_mfcc_features."""

    def test_default_shape(self, sine_wave: np.ndarray) -> None:
        feat = compute_mfcc_features(sine_wave, SR)
        # 40 means + 40 stds = 80
        assert feat.shape == (80,)

    def test_custom_n_mfcc(self, sine_wave: np.ndarray) -> None:
        feat = compute_mfcc_features(sine_wave, SR, n_mfcc=20)
        assert feat.shape == (40,)

    def test_returns_float32(self, sine_wave: np.ndarray) -> None:
        feat = compute_mfcc_features(sine_wave, SR)
        assert feat.dtype == np.float32

    def test_no_nans(self, sine_wave: np.ndarray) -> None:
        feat = compute_mfcc_features(sine_wave, SR)
        assert not np.any(np.isnan(feat))


class TestSegmentFixed:
    """Tests for segment_fixed."""

    def test_clip_length(self, sine_wave: np.ndarray) -> None:
        clips = segment_fixed(sine_wave, SR, clip_duration=1.0, overlap=0.0)
        expected_samples = int(SR * 1.0)
        for clip in clips:
            assert len(clip) == expected_samples

    def test_all_clips_correct_length_with_overlap(
        self, sine_wave: np.ndarray,
    ) -> None:
        clips = segment_fixed(sine_wave, SR, clip_duration=1.0, overlap=0.5)
        expected_samples = int(SR * 1.0)
        for clip in clips:
            assert len(clip) == expected_samples

    def test_number_of_clips_no_overlap(self, sine_wave: np.ndarray) -> None:
        clip_dur = 1.0
        clips = segment_fixed(sine_wave, SR, clip_duration=clip_dur, overlap=0.0)
        # 3 s / 1 s = 3 clips exactly
        assert len(clips) == int(DURATION / clip_dur)

    def test_last_clip_zero_padded(self) -> None:
        """A 2.5 s signal split into 1 s clips (no overlap) → 3 clips; last
        is 0.5 s of signal + 0.5 s of zeros."""
        short = np.ones(int(SR * 2.5), dtype=np.float32)
        clips = segment_fixed(short, SR, clip_duration=1.0, overlap=0.0)
        assert len(clips) == 3
        last = clips[-1]
        assert len(last) == SR  # still full clip length
        # Tail half should be all zeros (padding)
        assert np.all(last[int(SR * 0.5):] == 0.0)
