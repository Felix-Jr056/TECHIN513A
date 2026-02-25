# Fox Audio Detection

**Binary audio classification**: Detect whether an audio clip contains a fox vocalization or not.

## Project Goal

This project builds an end-to-end pipeline for **fox vs. non-fox audio classification**. It covers:

1. **Data collection** — Downloading and organizing raw audio files of fox calls and non-fox environmental sounds.
2. **Preprocessing** — Segmenting long recordings into fixed-length (3-second) clips, extracting MFCC features, and generating log-mel spectrogram images.
3. **Baseline model** — Training a traditional ML classifier (e.g., Random Forest / SVM) on MFCC feature vectors using scikit-learn.
4. **CNN model** — Training a convolutional neural network on log-mel spectrogram images using PyTorch.
5. **Evaluation** — Comparing both approaches with accuracy, precision, recall, F1-score, and confusion matrices.
6. **Demo** — An interactive Gradio web app for real-time fox audio detection.

---

## Directory Structure

```
fox_detection/
├── data/
│   ├── raw/
│   │   ├── fox/              # Raw downloaded fox audio files
│   │   └── nonfox/           # Raw downloaded non-fox audio files
│   ├── clips/
│   │   ├── fox/              # 3-second segmented fox clips
│   │   └── nonfox/           # 3-second segmented non-fox clips
│   ├── features/             # Saved .npy MFCC feature arrays
│   ├── spectrograms/         # Saved log-mel spectrogram .png images
│   │   ├── fox/
│   │   └── nonfox/
│   └── manifest.csv          # File ID, local path, source, label
├── models/
│   ├── baseline/             # Saved scikit-learn model (.pkl)
│   └── cnn/                  # Saved PyTorch model checkpoints (.pth)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_cnn_model.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── audio_utils.py        # Audio loading, resampling, normalization
│   ├── segmentation.py       # Splitting audio into fixed-length clips
│   ├── features.py           # MFCC extraction & spectrogram generation
│   ├── dataset.py            # PyTorch Dataset class
│   ├── baseline_model.py     # scikit-learn baseline training/inference
│   ├── cnn_model.py          # CNN architecture definition
│   ├── train_cnn.py          # CNN training loop
│   ├── evaluate.py           # Evaluation metrics & visualization
│   └── demo.py               # Gradio demo application
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate fox_detection
```

### Option 2: pip

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run Each Stage

### Stage 1: Data Exploration

Open and run `notebooks/01_data_exploration.ipynb` to:
- Inspect the raw audio files and their metadata
- Visualize waveforms and basic audio statistics
- Review the data manifest (`data/manifest.csv`)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Stage 2: Preprocessing

Open and run `notebooks/02_preprocessing.ipynb` to:
- Segment raw audio into 3-second clips → `data/clips/`
- Extract MFCC features → `data/features/`
- Generate log-mel spectrograms → `data/spectrograms/`

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

Or run from the command line:
```bash
python -m src.segmentation       # Segment audio clips
python -m src.features           # Extract features & spectrograms
```

### Stage 3: Baseline Model

Open and run `notebooks/03_baseline_model.ipynb` to:
- Load MFCC features
- Train a Random Forest / SVM classifier
- Save the model to `models/baseline/`

```bash
jupyter notebook notebooks/03_baseline_model.ipynb
```

Or run from the command line:
```bash
python -m src.baseline_model
```

### Stage 4: CNN Model

Open and run `notebooks/04_cnn_model.ipynb` to:
- Load spectrogram images via the custom PyTorch Dataset
- Define and train the CNN
- Save checkpoints to `models/cnn/`

```bash
jupyter notebook notebooks/04_cnn_model.ipynb
```

Or run from the command line:
```bash
python -m src.train_cnn
```

### Stage 5: Evaluation

Open and run `notebooks/05_evaluation.ipynb` to:
- Load both trained models
- Compare accuracy, precision, recall, F1-score
- Generate confusion matrices and ROC curves

```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

### Stage 6: Demo

Launch the interactive Gradio web app:

```bash
python -m src.demo
```

Upload an audio file and the app will predict whether it contains a fox vocalization.

---

## Source Modules

| Module | Description |
|---|---|
| `src/audio_utils.py` | Audio I/O, resampling, normalization utilities |
| `src/segmentation.py` | Split long recordings into fixed-length clips |
| `src/features.py` | Extract MFCCs and generate log-mel spectrograms |
| `src/dataset.py` | PyTorch `Dataset` for spectrogram images |
| `src/baseline_model.py` | Train and evaluate scikit-learn baseline |
| `src/cnn_model.py` | CNN architecture (PyTorch `nn.Module`) |
| `src/train_cnn.py` | Training loop for the CNN |
| `src/evaluate.py` | Metrics computation and visualization |
| `src/demo.py` | Gradio interactive demo |

---

## License

This project is for educational purposes (TECHIN 513A).
