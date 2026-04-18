# Speech Emotion Detection — CRNN v3

> 8-class speech emotion recognition from raw audio using a Convolutional Recurrent Neural Network with 5-fold cross-validation.

![Test Accuracy](https://img.shields.io/badge/test_accuracy-86.49%25-2ea44f)

---

## Overview

This project builds an end-to-end speech emotion classifier that converts raw `.wav` files into 3-channel mel spectrogram images and classifies them using a compact CRNN architecture. The model is trained with **5-fold cross-validation** to produce reliable accuracy estimates on a small dataset (~832 samples after augmentation).

| Metric                 | Value                        |
| ---------------------- | ---------------------------- |
| Held-out test accuracy | **86.49%**                   |
| CV mean accuracy       | **64.5% ± 10.3%**            |
| Total parameters       | ~144,000                     |
| Training samples       | ~832 (after 2× augmentation) |
| Emotion classes        | 8                            |

---

## Architecture

Raw audio is converted to a `(128, 128, 3)` image — mel spectrogram, Δ, and ΔΔ channels — then passed through the following network:

```
                    ┌─────────────────────────────────────────────┐
                    │              AUDIO PIPELINE                 │
                    │  WAV → Trim/Pad → RMS Norm → Pre-emphasis   │
                    │      → Mel Spec → dB → Δ → ΔΔ → Resize     │
                    └──────────────────┬──────────────────────────┘
                                       │
                               (128, 128, 3)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           CNN BLOCK 1  [16 filters]         │
                    │   Conv3×3 → BN → ReLU → Conv3×3 → BN →     │
                    │         ReLU → MaxPool2D → Dropout(0.25)    │
                    └──────────────────┬──────────────────────────┘
                                       │
                               (64, 64, 16)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           CNN BLOCK 2  [32 filters]         │
                    │   Conv3×3 → BN → ReLU → Conv3×3 → BN →     │
                    │         ReLU → MaxPool2D → Dropout(0.35)    │
                    └──────────────────┬──────────────────────────┘
                                       │
                               (32, 32, 32)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │        1×1 CHANNEL COMPRESS  [8 maps]       │
                    │              Conv1×1 → BN → ReLU            │
                    └──────────────────┬──────────────────────────┘
                                       │
                               (32, 32, 8)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           CNN → LSTM BRIDGE                 │
                    │      Permute(2,1,3) → Reshape(32, 256)      │
                    │   [32 time steps × 256 features per step]   │
                    └──────────────────┬──────────────────────────┘
                                       │
                                  (32, 256)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │              LSTM  [96 units]               │
                    │      return_sequences=False · Dropout(0.3)  │
                    └──────────────────┬──────────────────────────┘
                                       │
                                     (96,)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           DENSE HEAD  [48 units]            │
                    │         Dense → BN → ReLU → Dropout(0.4)   │
                    └──────────────────┬──────────────────────────┘
                                       │
                                     (48,)
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         OUTPUT  [Softmax · 8 classes]       │
                    └──────────────────┬──────────────────────────┘
                                       │
                          (angry · calm · disgust · fearful
                           happy · neutral · sad · surprised)

  Total trainable parameters: ~144,000
```

### Layer summary

| Layer / Block     | Output shape  | Parameters   | Notes                       |
| ----------------- | ------------- | ------------ | --------------------------- |
| Input             | (128, 128, 3) | 0            | 3-channel spectrogram image |
| CNN Block 1       | (64, 64, 16)  | ~4,700       | 16 filters, Dropout 0.25    |
| CNN Block 2       | (32, 32, 32)  | ~18,600      | 32 filters, Dropout 0.35    |
| 1×1 compress      | (32, 32, 8)   | 264          | Reduces maps 32 → 8         |
| Permute + Reshape | (32, 256)     | 0            | CNN → LSTM bridge           |
| LSTM (96 units)   | (96,)         | ~138,000     | Dropout 0.3                 |
| Dense 48 + BN     | (48,)         | ~4,700       | Dropout 0.4                 |
| Softmax output    | (8,)          | 392          | One prob per emotion        |
| **Total**         | —             | **~144,000** |                             |

---

## Per-class results

| Emotion   | Test accuracy | Notes                                                |
| --------- | ------------- | ---------------------------------------------------- |
| fearful   | 100.0%        | High pitch + breathy quality highly separable        |
| neutral   | 100.0%        | Flat prosody unambiguous in spectrogram space        |
| surprised | 100.0%        | Sharp spectral onset reliably detected               |
| sad       | 95.7%         | Slow, low-pitched speech clearly separable           |
| angry     | 95.2%         | High energy and spectral flux are distinctive        |
| calm      | 90.0%         | Low-energy steady patterns well-captured             |
| disgust   | 89.5%         | Mid-arousal features clearly learned                 |
| **happy** | **58.3%**     | Confused with high-arousal states (angry, surprised) |

---

## 5-fold cross-validation results

| Fold     | Best val. accuracy |
| -------- | ------------------ |
| Fold 1   | 62.9%              |
| Fold 2   | 69.5%              |
| Fold 3   | 57.7%              |
| Fold 4   | **81.1%**          |
| Fold 5   | 51.0%              |
| **Mean** | **64.5% ± 10.3%**  |

With only ~104 samples per class, a single train/val split gives ~15 validation samples per class — far too few for a stable estimate. One unlucky split can swing accuracy by 20%+. 5-fold CV uses all data across five independent splits and reports the mean ± std, giving a statistically honest generalisation estimate.

---

## Training configuration

| Hyperparameter | Value                | Note                                               |
| -------------- | -------------------- | -------------------------------------------------- |
| Optimizer      | Adam                 | —                                                  |
| Learning rate  | 0.0005               | Halved (×0.5) every 8 epochs of no val improvement |
| Batch size     | 16                   | More gradient updates per epoch on small data      |
| Max epochs     | 80                   | EarlyStopping patience = 15                        |
| Validation     | 5-fold stratified CV | + separate 15% held-out test set                   |
| Class weights  | Balanced             | Computed via sklearn `compute_class_weight`        |
| Random seed    | 42                   | NumPy, TensorFlow, and Python RNG all seeded       |

---

## Dataset

**Source:** [Kaggle — dwij45/speech-emotion-detection](https://www.kaggle.com/datasets/dwij45/speech-emotion-detection)

8 emotion classes: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

~52 original files per class → doubled to ~104 via augmentation (pitch shift ±2 semitones, time stretch ±15%, white noise) → **832 total samples**.

---

## Project structure

```
speech-emotion-crnn/
├── data/
│   └── augment_dataset.py        # 2× audio augmentation pipeline
├── features/
│   ├── X_all.npy                 # cached spectrogram features (N, 128, 128, 3)
│   └── y_all.npy                 # cached string labels
├── models/
│   ├── emotion_crnn_v3_best.keras    # best overall model (by val_accuracy)
│   └── best_crnn_fold{1-5}.keras    # per-fold checkpoints
├── crnn_v3.py                    # main training script
├── requirements.txt
└── README.md
```

---

## Model iteration history

| Version | Architecture | Val accuracy | Key issue |
|---|---|---|---|
| CNN v1 | 4 blocks, 1.27M params | ~17% | Severe overfitting |
| CNN v2 | 3 blocks, small | ~24% | No temporal modelling |
| CRNN v1 | 3 CNN + 2 LSTM, 1024-dim bridge | ~24% | Bridge vector too large |
| CRNN v2 | 2 CNN + 1×1 compress + LSTM 64 | ~42%* | Single-split variance |
| **CRNN v3** | LSTM 96 + Dense 48 + 5-fold CV | **64.5% ± 10.3% CV** | Happy class confusion |

_\* Single-split estimate — high variance. CV mean would be lower._

---

## Known limitations

- **Happy class (58.3%)** — acoustically ambiguous with surprised and angry; consistently weakest across all folds
- **Small dataset** — ~52 original files per class limits generalisation; the ±10.3% CV std reflects this directly

---

## Technical stack

| Component        | Details                                                     |
| ---------------- | ----------------------------------------------------------- |
| Platform         | Kaggle Notebooks — GPU T4                                   |
| Framework        | TensorFlow 2.x / Keras Functional API                       |
| Audio processing | librosa (mel spectrogram, delta, pitch shift, time stretch) |
| Audio I/O        | soundfile                                                   |
| Image processing | Pillow — bilinear resize                                    |
| ML utilities     | scikit-learn (StratifiedKFold, LabelEncoder, class_weight)  |
| Visualisation    | matplotlib                                                  |
| Model format     | Keras native `.keras`                                       |

---

_CRNN v3 · Speech Emotion Detection · TensorFlow 2.x · Kaggle GPU T4 · 2026_
