"""
demo.py  –  🦊 Red Fox Vocalisation Detector
Gradio web app that lets users upload / record audio and get per-clip
predictions from either a CNN (EfficientNet-B0) or a baseline SVM model.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from PIL import Image

# ── Dark tech plot style (matches GitHub Pages) ─────────────────────────────
_CYAN = "#00e5ff"
_PURPLE = "#7c4dff"
_GREEN = "#00e676"
_BG = "#000000"
_SURFACE = "#0a0a0a"
_TEXT = "#e0e0e0"
_MUTED = "#666666"
_FONT = "monospace"

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor": _SURFACE,
    "axes.edgecolor": _MUTED,
    "axes.labelcolor": _TEXT,
    "text.color": _TEXT,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "grid.color": "#1a1a1a",
    "font.family": _FONT,
    "font.size": 10,
    "savefig.facecolor": _BG,
})

from src.audio_utils import (
    compute_log_mel_spectrogram,
    compute_mfcc_features,
    load_audio,
    normalise_audio,
    save_spectrogram_image,
)
from src.baseline_model import BaselineClassifier
from src.cnn_model import FoxCNN
from src.segmentation import segment_fixed

# ── Module-level model cache ─────────────────────────────────────────────────

_MODEL_CACHE: dict[str, Any] = {}

SR = 22050
CLIP_DURATION = 3.0
OVERLAP = 0.5
FOX_THRESHOLD = 0.30          # >30 % of clips → file-level "FOX"
IMG_SIZE = (128, 128)          # CNN input size


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(
    cnn_path: str | None = None,
    baseline_path: str | None = None,
) -> None:
    """Load models once and store them in ``_MODEL_CACHE``."""
    if cnn_path and "cnn" not in _MODEL_CACHE:
        dev = _get_device()
        ckpt = torch.load(cnn_path, map_location=dev, weights_only=False)
        backbone = ckpt.get("backbone", "efficientnet_b0")
        model = FoxCNN(backbone=backbone, pretrained=False, num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev).eval()
        _MODEL_CACHE["cnn"] = model
        _MODEL_CACHE["cnn_device"] = dev
        print(f"[demo] CNN loaded from {cnn_path}  (device={dev})")

    if baseline_path and "baseline" not in _MODEL_CACHE:
        clf = BaselineClassifier.load(baseline_path)
        _MODEL_CACHE["baseline"] = clf
        print(f"[demo] Baseline loaded from {baseline_path}")


# ── Per-clip inference helpers ────────────────────────────────────────────────

def _predict_clip_cnn(clip: np.ndarray) -> tuple[int, float]:
    """Return (label, confidence) for one clip using CNN."""
    from torchvision import transforms as T

    model = _MODEL_CACHE["cnn"]
    dev = _MODEL_CACHE["cnn_device"]

    # Save clip as spectrogram image → load as tensor
    spec = compute_log_mel_spectrogram(clip, SR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        tmp = fp.name
    save_spectrogram_image(spec, tmp)
    img = Image.open(tmp).convert("RGB").resize(IMG_SIZE)
    os.remove(tmp)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    fox_prob = probs[1].item()
    label = 1 if fox_prob >= 0.5 else 0
    return label, fox_prob


def _predict_clip_baseline(clip: np.ndarray) -> tuple[int, float]:
    """Return (label, confidence) for one clip using baseline SVM."""
    clf: BaselineClassifier = _MODEL_CACHE["baseline"]
    feat = compute_mfcc_features(clip, SR).reshape(1, -1)
    label = int(clf.pipeline.predict(feat)[0])
    if hasattr(clf.pipeline, "predict_proba"):
        fox_prob = float(clf.pipeline.predict_proba(feat)[0, 1])
    else:
        fox_prob = float(clf.pipeline.decision_function(feat)[0])
    return label, fox_prob


# ── Visualization helpers ─────────────────────────────────────────────────────

def _plot_waveform_with_clips(
    y: np.ndarray,
    sr: int,
    clips: list[np.ndarray],
    labels: list[int],
    confidences: list[float],
) -> plt.Figure:
    """Draw full waveform with clip-boundary colour coding."""
    clip_samples = int(sr * CLIP_DURATION)
    hop_samples = int(sr * (CLIP_DURATION - OVERLAP))

    fig, ax = plt.subplots(figsize=(12, 3))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, color=_CYAN, linewidth=0.4, alpha=0.7)

    for idx, (lab, conf) in enumerate(zip(labels, confidences)):
        start_s = idx * hop_samples / sr
        end_s = start_s + CLIP_DURATION
        colour = _PURPLE if lab == 1 else "#1a1a1a"
        alpha = 0.40 if lab == 1 else 0.20
        ax.axvspan(start_s, min(end_s, t[-1]), color=colour, alpha=alpha)
        # small label on top
        mid = (start_s + min(end_s, t[-1])) / 2
        emoji = "🦊" if lab == 1 else ""
        ax.text(mid, ax.get_ylim()[1] * 0.85, f"{emoji}{conf:.0%}",
                ha="center", fontsize=7, color=_CYAN if lab == 1 else _MUTED)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform — per-clip predictions (purple = fox)", color=_TEXT)
    fig.tight_layout()
    return fig


def _plot_spectrogram(clip: np.ndarray, sr: int, title: str) -> plt.Figure:
    """Plot a log-mel spectrogram of a single clip."""
    spec = compute_log_mel_spectrogram(clip, sr)
    fig, ax = plt.subplots(figsize=(5, 3))
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno",
                    extent=[0, CLIP_DURATION, 0, sr // 2])
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_MUTED)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, color=_TEXT)
    fig.tight_layout()
    return fig


# ── Main prediction pipeline ─────────────────────────────────────────────────

def predict(
    audio_filepath: str,
    model_choice: str = "CNN (EfficientNet-B0)",
) -> tuple[str, str, plt.Figure, plt.Figure]:
    """Run full inference pipeline on an uploaded audio file.

    Parameters
    ----------
    audio_filepath : str
        Path to the user-uploaded audio file.
    model_choice : str
        ``"CNN (EfficientNet-B0)"`` or ``"Baseline SVM"``.

    Returns
    -------
    label_text : str
        Prediction label with emoji.
    confidence_text : str
        Confidence score as a string.
    waveform_fig : matplotlib.figure.Figure
        Waveform plot with clip annotations.
    spec_fig : matplotlib.figure.Figure
        Spectrogram of the most-confident fox clip (or first clip).
    """
    # 1. Load & standardise
    y, sr = load_audio(audio_filepath, target_sr=SR)
    y = normalise_audio(y)

    # 2. Segment into 3-second clips
    clips = segment_fixed(y, sr, clip_duration=CLIP_DURATION, overlap=OVERLAP)

    # 3. Per-clip inference
    use_cnn = "cnn" in model_choice.lower()
    clip_fn = _predict_clip_cnn if use_cnn else _predict_clip_baseline

    labels: list[int] = []
    confidences: list[float] = []
    for clip in clips:
        lab, conf = clip_fn(clip)
        labels.append(lab)
        confidences.append(conf)

    # 4. Aggregate: FOX if >30 % clips are fox
    n_fox = sum(labels)
    fox_ratio = n_fox / len(labels) if labels else 0.0
    is_fox = fox_ratio > FOX_THRESHOLD

    if is_fox:
        avg_conf = np.mean([c for c, l in zip(confidences, labels) if l == 1])
        label_text = "FOX DETECTED"
    else:
        avg_conf = 1.0 - np.mean(confidences)
        label_text = "No fox detected"
    confidence_text = f"{avg_conf:.1%}"

    # 5a. Waveform figure
    waveform_fig = _plot_waveform_with_clips(y, sr, clips, labels, confidences)

    # 5b. Spectrogram of the most-confident fox clip (or first clip)
    fox_indices = [i for i, l in enumerate(labels) if l == 1]
    if fox_indices:
        best_idx = max(fox_indices, key=lambda i: confidences[i])
        spec_title = f"Most confident fox clip (#{best_idx}, {confidences[best_idx]:.1%})"
    else:
        best_idx = 0
        spec_title = "First clip (no fox detected)"
    spec_fig = _plot_spectrogram(clips[best_idx], sr, spec_title)

    return label_text, confidence_text, waveform_fig, spec_fig


# ── Gradio interface ──────────────────────────────────────────────────────────

def launch_demo(
    cnn_path: str | None = None,
    baseline_path: str | None = None,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Build and launch the Gradio app."""
    import gradio as gr

    # Load models at startup
    load_models(cnn_path=cnn_path, baseline_path=baseline_path)

    # Build list of available models based on what was loaded
    choices: list[str] = []
    if "cnn" in _MODEL_CACHE:
        choices.append("CNN (EfficientNet-B0)")
    if "baseline" in _MODEL_CACHE:
        choices.append("Baseline SVM")
    if not choices:
        raise RuntimeError(
            "No models loaded — provide at least --cnn_model or --baseline_model"
        )

    # ── Custom dark-tech CSS matching GitHub Pages theme ──────────────────
    custom_css = """
    :root {
        --body-background-fill: #000000 !important;
    }
    .gradio-container {
        background: #000000 !important;
        font-family: 'SF Mono','Fira Code','Cascadia Code','JetBrains Mono',monospace !important;
        max-width: 1100px !important;
    }
    .gr-box, .gr-panel, .gr-form {
        background: #0a0a0a !important;
        border-color: #1a1a1a !important;
    }
    .gr-button.gr-button-primary {
        background: #00e5ff !important;
        color: #000 !important;
        border: none !important;
        font-weight: 700 !important;
        font-family: 'SF Mono','Fira Code',monospace !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .gr-button.gr-button-primary:hover {
        background: #00b8d4 !important;
    }
    h1, .markdown-text h1 {
        color: #00e5ff !important;
        font-family: 'SF Mono','Fira Code',monospace !important;
        font-weight: 300 !important;
        letter-spacing: -1px;
    }
    .markdown-text, .markdown-text p, label, .gr-input-label {
        color: #e0e0e0 !important;
        font-family: 'SF Mono','Fira Code',monospace !important;
    }
    textarea, input, .gr-text-input {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
        border-color: #1a1a1a !important;
        font-family: 'SF Mono','Fira Code',monospace !important;
    }
    .gr-dropdown {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
        border-color: #1a1a1a !important;
    }
    footer { display: none !important; }
    .label-wrap, .icon {
        color: #666666 !important;
    }
    """

    dark_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e0f7fa", c100="#b2ebf2", c200="#80deea",
            c300="#4dd0e1", c400="#26c6da", c500="#00e5ff",
            c600="#00b8d4", c700="#0097a7", c800="#00838f",
            c900="#006064", c950="#004d40",
        ),
        secondary_hue=gr.themes.Color(
            c50="#ede7f6", c100="#d1c4e9", c200="#b39ddb",
            c300="#9575cd", c400="#7e57c2", c500="#7c4dff",
            c600="#651fff", c700="#6200ea", c800="#4a148c",
            c900="#311b92", c950="#1a0066",
        ),
        neutral_hue=gr.themes.Color(
            c50="#e0e0e0", c100="#bdbdbd", c200="#9e9e9e",
            c300="#757575", c400="#616161", c500="#424242",
            c600="#2a2a2a", c700="#1a1a1a", c800="#111111",
            c900="#0a0a0a", c950="#000000",
        ),
    ).set(
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        block_background_fill="#0a0a0a",
        block_background_fill_dark="#0a0a0a",
        block_border_color="#1a1a1a",
        block_border_color_dark="#1a1a1a",
        block_label_text_color="#666666",
        block_label_text_color_dark="#666666",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
        input_background_fill="#0a0a0a",
        input_background_fill_dark="#0a0a0a",
        input_border_color="#1a1a1a",
        input_border_color_dark="#1a1a1a",
        border_color_primary="#1a1a1a",
        border_color_primary_dark="#1a1a1a",
    )

    with gr.Blocks(
        title="Fox Audio Detection",
    ) as demo:
        gr.Markdown("# Fox **Audio Detection**")
        gr.Markdown(
            "Upload or record audio — the model classifies whether it "
            "contains a **red fox** vocalisation."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="Model",
                )
                analyse_btn = gr.Button("ANALYSE", variant="primary")

            with gr.Column(scale=2):
                label_output = gr.Textbox(label="Prediction", interactive=False)
                conf_output = gr.Textbox(label="Confidence", interactive=False)
                waveform_plot = gr.Plot(label="Waveform & clip predictions")
                spec_plot = gr.Plot(label="Log-mel spectrogram")

        analyse_btn.click(
            fn=predict,
            inputs=[audio_input, model_dropdown],
            outputs=[label_output, conf_output, waveform_plot, spec_plot],
        )

    demo.launch(server_port=port, share=share, theme=dark_theme, css=custom_css)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🦊 Red Fox Vocalisation Detector — Gradio demo"
    )
    parser.add_argument(
        "--cnn_model", type=str, default=None,
        help="Path to CNN checkpoint (.pt).",
    )
    parser.add_argument(
        "--baseline_model", type=str, default=None,
        help="Path to baseline model (.pkl).",
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifest.csv",
        help="Path to manifest CSV (unused at runtime; kept for CLI consistency).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Server port (default 7860).",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    args = parser.parse_args()

    launch_demo(
        cnn_path=args.cnn_model,
        baseline_path=args.baseline_model,
        port=args.port,
        share=args.share,
    )
