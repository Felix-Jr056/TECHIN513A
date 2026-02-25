"""
audio_utils.py
Audio loading, resampling, normalisation, feature-extraction, and
spectrogram utilities for the fox-detection project.
"""

from __future__ import annotations

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt


# ── 1. Loading ────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load any audio file, resample to *target_sr*, and convert to mono.

    Parameters
    ----------
    path : str
        Path to the audio file (wav, mp3, flac, ogg, …).
    target_sr : int, optional
        Target sample rate in Hz (default 22 050).

    Returns
    -------
    y : np.ndarray
        1-D mono waveform, dtype float32.
    sr : int
        The sample rate (== *target_sr*).
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr


# ── 2. Normalisation ─────────────────────────────────────────────────────────

def normalise_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalise a waveform to the range [-1, 1].

    Parameters
    ----------
    y : np.ndarray
        Input waveform.

    Returns
    -------
    np.ndarray
        Peak-normalised waveform.  Returns zeros if the signal is silent.
    """
    peak = np.max(np.abs(y))
    if peak == 0.0:
        return y
    return (y / peak).astype(np.float32)


# ── 3. Silence trimming ──────────────────────────────────────────────────────

def trim_silence(y: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
    """Remove leading and trailing silence using librosa.

    Parameters
    ----------
    y : np.ndarray
        Input waveform.
    sr : int
        Sample rate (used internally by librosa).
    top_db : float, optional
        Threshold (in dB) below the signal peak to consider as silence
        (default 30).

    Returns
    -------
    np.ndarray
        Trimmed waveform.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


# ── 4. Log-mel spectrogram ───────────────────────────────────────────────────

def compute_log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmax: float = 8000,
) -> np.ndarray:
    """Compute a log-mel spectrogram in dB.

    Parameters
    ----------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    n_mels : int, optional
        Number of Mel bands (default 128).
    hop_length : int, optional
        Hop length in samples (default 512).
    n_fft : int, optional
        FFT window size (default 2048).
    fmax : float, optional
        Maximum frequency for the Mel filterbank (default 8 000 Hz).

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, time_frames)`` log-mel spectrogram in dB.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft, fmax=fmax,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


# ── 5. Spectrogram image saving ──────────────────────────────────────────────

def save_spectrogram_image(
    spec: np.ndarray,
    save_path: str,
    figsize: tuple[int, int] = (3, 3),
) -> None:
    """Save a spectrogram as a fixed-size grayscale PNG (no axes/labels).

    The image is suitable for direct use as CNN input.

    Parameters
    ----------
    spec : np.ndarray
        2-D spectrogram array, e.g. from :func:`compute_log_mel_spectrogram`.
    save_path : str
        Destination file path (PNG).
    figsize : tuple[int, int], optional
        Figure size in inches (default ``(3, 3)``).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(spec, aspect="auto", origin="lower", cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


# ── 6. MFCC feature vector ───────────────────────────────────────────────────

def compute_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Compute a fixed-length MFCC feature vector (mean + std per coefficient).

    Parameters
    ----------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    n_mfcc : int, optional
        Number of MFCC coefficients (default 40).

    Returns
    -------
    np.ndarray
        1-D feature vector of length ``2 * n_mfcc`` (means then stds).
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, T)
    means = np.mean(mfccs, axis=1)
    stds  = np.std(mfccs, axis=1)
    return np.concatenate([means, stds]).astype(np.float32)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    SR = 22050
    DURATION = 3.0  # seconds
    FREQ = 440.0    # A4

    # Synthesise a 3-second 440 Hz sine wave
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * FREQ * t)).astype(np.float32)

    print("=== audio_utils self-test ===\n")

    # normalise
    normed = normalise_audio(sine)
    assert np.isclose(np.max(np.abs(normed)), 1.0), "normalise_audio failed"
    print(f"✓ normalise_audio  – peak={np.max(np.abs(normed)):.4f}")

    # trim silence (pad with zeros then trim)
    padded = np.concatenate([np.zeros(SR), sine, np.zeros(SR)])
    trimmed = trim_silence(padded, SR, top_db=30)
    assert len(trimmed) < len(padded), "trim_silence did not remove silence"
    print(f"✓ trim_silence     – {len(padded)} → {len(trimmed)} samples")

    # log-mel spectrogram
    spec = compute_log_mel_spectrogram(sine, SR)
    assert spec.shape[0] == 128, f"Expected 128 mel bands, got {spec.shape[0]}"
    print(f"✓ log_mel_spec     – shape {spec.shape}")

    # save spectrogram image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    save_spectrogram_image(spec, tmp_path)
    assert os.path.isfile(tmp_path), "spectrogram image not saved"
    fsize = os.path.getsize(tmp_path)
    print(f"✓ save_spec_image  – {tmp_path} ({fsize} bytes)")
    os.remove(tmp_path)

    # MFCC features
    feat = compute_mfcc_features(sine, SR, n_mfcc=40)
    assert feat.shape == (80,), f"Expected (80,), got {feat.shape}"
    print(f"✓ mfcc_features    – shape {feat.shape}")

    print("\n=== all tests passed ===")
