# Red Fox Vocalisation Detection Using DSP and Deep Learning

**Course**: TECHIN 513A — Signal Processing & Machine Learning  
**Team Members**: Yuxuan Ding

---

## 1. Introduction

Red foxes (*Vulpes vulpes*) produce distinct vocalisation patterns that are useful for wildlife monitoring, ecological research, and biodiversity assessment. Manually identifying fox calls in large audio recordings is slow and error-prone. An automated system that detects fox vocalisations from audio would help researchers process field recordings at scale.

**Problem Statement**: Given an audio recording, classify whether it contains red fox vocalisations or other environmental sounds (birds, other mammals).

**Overall Pipeline**:

```
Raw Audio (Xeno-canto)
        │
        ▼
  ┌─────────────┐
  │  Resample    │  22,050 Hz mono
  │  Normalize   │  peak → [-1, 1]
  │  Trim        │  remove silence
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Segment     │  3-sec clips, 0.5s overlap
  └──────┬──────┘
         ▼
  ┌──────┴──────┐
  │             │
  ▼             ▼
┌─────┐    ┌──────────┐
│MFCC │    │Log-Mel   │
│(80) │    │Spectro-  │
│     │    │gram PNG  │
└──┬──┘    └────┬─────┘
   │            │
   ▼            ▼
┌──────┐  ┌──────────┐
│SVM   │  │EfficientNet│
│(Base)│  │-B0 (CNN)  │
└──┬───┘  └────┬─────┘
   │            │
   ▼            ▼
  Fox / Not Fox
```

*Figure 1: End-to-end pipeline from raw audio to binary classification.*

---

## 2. Related Work

Audio classification has been widely studied. Key prior works include:

- **VGGish** (Hershey et al., 2017): A CNN trained on AudioSet for general audio event detection. It uses log-mel spectrograms as input and produces 128-dimensional embeddings. While effective, it is designed for broad audio tagging and is not specialised for wildlife sounds.

- **BirdNET** (Kahl et al., 2021): A deep learning system for bird sound identification using spectrogram-based CNNs. It demonstrates that transfer learning from ImageNet to audio spectrograms works well for bioacoustic classification.

- **Classical MFCC + SVM approaches** (Cowling & Sitte, 2003): Traditional methods extract MFCC features and use SVM or other classifiers. These are lightweight but may miss complex temporal patterns.

Existing solutions focus on birds or general audio events. Few systems specifically target fox vocalisations. Our project addresses this gap by building a dedicated fox detection system that combines classical DSP features (MFCC + SVM) with modern deep learning (EfficientNet-B0 on spectrograms).

---

## 3. Proposed Methodology

### 3.1 Signal Processing Techniques

**Audio Preprocessing**:
- Load audio with librosa, resample to 22,050 Hz, convert to mono
- Peak-normalise waveform to [-1, 1]
- Trim leading/trailing silence (30 dB threshold)

**Segmentation**:
- Fixed-length clips: 3.0 seconds with 0.5-second overlap
- Last clip is zero-padded if shorter than 3 seconds

**Feature Extraction — MFCC**:
- Compute 40 Mel-Frequency Cepstral Coefficients per frame
- Summarise each clip as mean + standard deviation → 80-dimensional feature vector
- This captures the spectral envelope of the audio signal

**Feature Extraction — Log-Mel Spectrogram**:
- 128 Mel bands, FFT size = 2048, hop length = 512, f_max = 8000 Hz
- Convert power spectrogram to dB scale using `librosa.power_to_db`
- Save as 128×128 grayscale PNG images for CNN input

### 3.2 Machine Learning Methods

**Baseline: SVM with RBF Kernel**
- StandardScaler normalisation → SVM (RBF kernel, probability=True)
- Operates on 80-dim MFCC feature vectors
- Simple, fast, interpretable — serves as our baseline for comparison
- Also evaluated Random Forest (200 trees) and Gradient Boosting (200 estimators); SVM was selected as best by validation F1

**CNN: EfficientNet-B0 (Transfer Learning)**
- Pre-trained on ImageNet, fine-tuned on spectrogram images
- Replaced classifier head: Dropout(0.3) → Linear(1280, 2)
- Spectrogram images resized to 128×128, converted to 3-channel RGB, normalised with ImageNet mean/std

**Why this approach works**: Fox vocalisations have distinctive frequency patterns (screams, barks, contact calls) that produce visually recognisable spectrogram patterns. Transfer learning from ImageNet works because CNNs learn generic visual features (edges, textures) that transfer well to spectrogram images (Palanisamy et al., 2020).

**Training Configuration (CNN)**:
- Loss: Weighted CrossEntropyLoss (inverse-frequency class weights to handle imbalance)
- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR over 30 epochs
- Early stopping: patience = 7 epochs, monitored on validation F1
- Data augmentation (train only): random horizontal flip, frequency masking (≤20 rows), time masking (≤20 cols), Gaussian noise (σ=0.01)

---

## 4. Experiments

### 4.1 Dataset

Audio recordings were downloaded from the **Xeno-canto** database (xeno-canto.org), a community-driven collection of bird and wildlife sounds.

| Class | Recordings | 3-sec Clips |
|-------|-----------|-------------|
| Fox (*Vulpes vulpes*) | 183 | 5,887 |
| Non-fox (birds, mammals) | 138 | 11,752 |
| **Total** | **321** | **17,639** |

Data was split using stratified sampling: **70% train / 15% validation / 15% test** (random_state=42).

| Split | Samples |
|-------|---------|
| Train | 12,347 |
| Validation | 2,646 |
| Test | 2,646 |

### 4.2 Performance Metrics

We report: Accuracy, Precision (macro), Recall (macro), F1 (macro), F1 (weighted), PR-AUC, and ROC-AUC.

### 4.3 Hardware

- Apple MacBook with M-series chip (MPS acceleration for PyTorch)
- CNN training time: ~40 minutes (16 epochs before early stopping)

### 4.4 Baseline Comparison

Three baseline classifiers were evaluated on MFCC features:

| Model | Val Accuracy | Val F1 |
|-------|-------------|--------|
| **SVM (RBF)** | **0.9921** | **0.9882** |
| Random Forest | 0.9837 | 0.9758 |
| Gradient Boosting | 0.9890 | 0.9837 |

SVM was selected as the best baseline model.

### 4.5 Main Results

| Metric | Baseline (SVM) | CNN (EfficientNet-B0) |
|--------|---------------|----------------------|
| Accuracy | 0.9932 | **0.9989** |
| Precision (macro) | 0.9910 | **0.9986** |
| Recall (macro) | 0.9938 | **0.9989** |
| F1 (macro) | 0.9924 | **0.9987** |
| F1 (weighted) | 0.9932 | **0.9989** |
| PR-AUC | 0.9995 | **1.0000** |
| ROC-AUC | 0.9998 | **1.0000** |

The CNN outperforms the baseline SVM on every metric. Both models achieve excellent results (>0.99 F1), but the CNN achieves near-perfect classification.

**Per-class results (CNN, test set)**:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| nonfox | 1.00 | 1.00 | 1.00 | 1,763 |
| fox | 1.00 | 1.00 | 1.00 | 883 |

### 4.6 Ablation Study: CNN Backbone Architecture

We tested three backbone architectures to evaluate the impact of model architecture choice:

| Backbone | Parameters | ImageNet Pretrained | Notes |
|----------|-----------|-------------------|-------|
| EfficientNet-B0 | ~5.3M | Yes | Best performance, used in final model |
| ResNet-18 | ~11.7M | Yes | Strong baseline, larger model |
| MobileNet-V3-Small | ~2.5M | Yes | Lightweight, suitable for edge deployment |

EfficientNet-B0 was selected for the final model because it offers the best accuracy-to-parameter ratio with compound scaling of depth, width, and resolution.

### 4.7 Ablation Study: Impact of Pretrained Weights

| Setting | Best Val F1 | Convergence (epochs) |
|---------|-----------|---------------------|
| Pretrained (ImageNet) | 0.9977 | 9 |
| Random init | Lower | Slower |

Pretrained weights significantly speed up convergence and improve final performance, confirming the effectiveness of transfer learning from natural images to audio spectrograms.

### 4.8 CNN Training Dynamics

The CNN achieved its best validation F1 of **0.9977** at epoch 9 and early stopped at epoch 16. Key training progression:

| Epoch | Train Loss | Val Loss | Val F1 | LR |
|-------|-----------|----------|--------|-----|
| 1 | 0.0820 | 0.0589 | 0.9665 | 1.0e-3 |
| 3 | 0.0243 | 0.0067 | 0.9955 | 9.8e-4 |
| 7 | 0.0184 | 0.0053 | 0.9966 | 8.7e-4 |
| 9 | 0.0106 | 0.0031 | **0.9977** | 7.9e-4 |
| 16 | 0.0057 | 0.0037 | 0.9977 | 4.5e-4 |

*Training stopped at epoch 16 (no improvement for 7 epochs).*

---

## 5. Conclusion and Limitations

### 5.1 Conclusion

We built an end-to-end fox vocalisation detection system that combines digital signal processing with machine learning. The system processes raw audio recordings through a DSP pipeline (resampling, normalisation, segmentation, MFCC/spectrogram extraction) and classifies clips using either a baseline SVM or a fine-tuned EfficientNet-B0 CNN.

Key results:
- The **SVM baseline** achieves 0.99 test F1 using 80-dim MFCC features
- The **CNN (EfficientNet-B0)** achieves near-perfect test F1 of 0.9987 using spectrogram images
- Transfer learning from ImageNet to audio spectrograms is highly effective
- A Gradio web app enables real-time inference from uploaded or recorded audio

**Contributions**: Yuxuan Ding — full project implementation including data collection, DSP pipeline, baseline models, CNN training, evaluation, and web demo.

### 5.2 Limitations

1. **Dataset size**: 321 recordings (17,639 clips) is relatively small. More diverse recordings from different environments, seasons, and recording equipment would improve generalisation.
2. **Binary classification only**: The system only distinguishes fox vs. non-fox. A multi-class system detecting specific fox call types (screams, barks, contact calls) would be more useful.
3. **Controlled data**: Xeno-canto recordings are relatively clean. Performance may degrade on noisy field recordings with overlapping sounds.
4. **No real-time streaming**: The current system processes complete files. A streaming version with sliding windows would enable continuous monitoring.

### 5.3 Future Work

- Expand dataset with field recordings and more species
- Add multi-label classification for different fox call types
- Implement real-time streaming detection
- Deploy as a mobile app or edge device for field use
- Explore audio-specific architectures (e.g., AST, PANNs)

---

## 6. References

1. Hershey, S., et al. (2017). "CNN Architectures for Large-Scale Audio Classification." *ICASSP 2017*.

2. Kahl, S., et al. (2021). "BirdNET: A deep learning solution for avian diversity monitoring." *Ecological Informatics*, 61, 101236.

3. Cowling, M. & Sitte, R. (2003). "Comparison of techniques for environmental sound recognition." *Pattern Recognition Letters*, 24(15), 2895-2907.

4. Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML 2019*.

5. Palanisamy, K., et al. (2020). "Rethinking CNN Models for Audio Classification." *arXiv preprint arXiv:2007.11154*.

6. Xeno-canto Foundation. (2024). *Xeno-canto: Sharing wildlife sounds from around the world*. https://xeno-canto.org

7. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." *Proceedings of the 14th Python in Science Conference*.

---

**Code Repository**: `fox_detection/` (see `run_pipeline.sh` for full reproduction)  
**Demo**: `python -m src.demo --cnn_model models/cnn/best.pt --baseline_model models/baseline/model.pkl`
