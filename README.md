# FractureVision v1.0
### Automated Forensic Glass Fracture Analysis System

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![Accuracy](https://img.shields.io/badge/Accuracy-80.8%25-brightgreen) ![AUC](https://img.shields.io/badge/AUC--ROC-0.9231-blue)

> An AI-powered forensic decision support tool for classifying glass fracture patterns and estimating the Point of Impact (POI) using deep learning and computer vision.

---

## Overview

FractureVision automatically classifies glass fracture photographs as:
- **High-Velocity** — bullet/gunshot impact
- **Low-Velocity** — blunt force impact (hammer, rock, etc.)

It also estimates the **Point of Impact (POI)** using the Probabilistic Hough Line Transform, generating a 4-panel forensic overlay suitable for investigative reporting and scene reconstruction.

---

## Features

- Binary fracture classification using fine-tuned **ResNet-50**
- Preprocessing pipeline: grayscale → Gaussian blur → Canny edge detection → morphological dilation
- Two-phase transfer learning (feature extraction + fine-tuning)
- Data augmentation to handle limited forensic datasets
- POI estimation with uncertainty scatter and concentric zone annotations
- Forensic-grade 4-panel visualization overlay

---

## Model Performance

| Metric | High-Velocity | Low-Velocity | Overall |
|--------|--------------|-------------|---------|
| Precision | 90.0% | 75.0% | — |
| Recall | 69.2% | 92.3% | — |
| F1-Score | 78.3% | 82.8% | — |
| Accuracy | — | — | **80.8%** |
| AUC-ROC | — | — | **0.9231** |

---

## Project Structure

```
FractureVision/
│
├── data/
│   ├── train/
│   │   ├── high_velocity/
│   │   └── low_velocity/
│   └── test/
│       ├── high_velocity/
│       └── low_velocity/
│
├── models/
│   └── fracturevision_resnet50.h5
│
├── src/
│   ├── preprocess.py        # Preprocessing pipeline
│   ├── train.py             # Two-phase training script
│   ├── evaluate.py          # Evaluation & metrics
│   └── poi_estimation.py    # Point of Impact module
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── poi_overlay.png
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FractureVision.git
cd FractureVision

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

```
tensorflow>=2.10
opencv-python>=4.5
scikit-learn>=1.0
numpy>=1.21
matplotlib>=3.5
seaborn>=0.11
```

---

## Usage

### 1. Preprocess Images
```python
from src.preprocess import preprocess_image

skeleton = preprocess_image("path/to/fracture_image.jpg")
```

### 2. Train the Model
```bash
python src/train.py --data_dir ./data --epochs_phase1 12 --epochs_phase2 30
```

### 3. Evaluate
```bash
python src/evaluate.py --model_path ./models/fracturevision_resnet50.h5 --test_dir ./data/test
```

### 4. Run POI Estimation
```bash
python src/poi_estimation.py --image_path ./data/test/sample.jpg
```

---

## Model Architecture

```
Input Image (224x224x3)
        ↓
  Preprocessing Pipeline
  (Grayscale → Blur → Canny → Dilation)
        ↓
  ResNet-50 Backbone (ImageNet weights)
  [Frozen in Phase 1 | Top 60 layers unfrozen in Phase 2]
        ↓
  Global Average Pooling
        ↓
  Dense(512) → Dropout(0.5)
  Dense(256) → Dropout(0.4)
  Dense(128) → Dropout(0.3)
        ↓
  SoftMax(2)
        ↓
  High-Velocity / Low-Velocity
```

---

## Training Strategy

| Phase | Layers Trained | Epochs | Learning Rate |
|-------|---------------|--------|--------------|
| Phase 1 — Feature Extraction | Classification head only | 12 | 1×10⁻³ |
| Phase 2 — Fine-Tuning | Top 60 ResNet layers + head | 30 | 5×10⁻⁶ |

- **Loss:** Sparse Categorical Cross-Entropy
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

---

## Dataset

~200 real forensic glass fracture photographs from controlled experimental scenarios:
- 77 training images — High-Velocity
- 79 training images — Low-Velocity
- 34 test images (17 per class) — held out until final evaluation

---

## Forensic Context

This tool is built around the **Locard Exchange Principle** — the axiom that every contact leaves a trace. In glass fractures, radial lines from a first impact terminate at those from subsequent impacts, enabling temporal sequencing of events.

> ⚠️ **Disclaimer:** FractureVision is a forensic *decision support* tool. It is not a replacement for certified forensic examination. All outputs should be reviewed and validated by a qualified forensic examiner before use in legal proceedings.

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| TensorFlow / Keras | Model training |
| OpenCV | Preprocessing & Hough transform |
| scikit-learn | Metrics & evaluation |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Visualization |
| Google Colab (T4 GPU) | Training environment |

---

## Author
Laiba Maqbool, Amna Saif

## References

1. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR 2016.*
2. Canny, J. "A Computational Approach to Edge Detection." *IEEE TPAMI, 1986.*
3. Locard, E. *L'enquête criminelle et les méthodes scientifiques.* 1920.
4. Delly, J. G. & Jiao, Y. "Forensic Glass Analysis." *Microscopy and Microanalysis, 2014.*
5. Simonyan, K. & Zisserman, A. "Very Deep Convolutional Networks." *ICLR 2015.*
6. OpenCV. https://opencv.org
7. TensorFlow. https://tensorflow.org

---

## License

This project was developed for academic purposes at Air University. Not licensed for commercial forensic use without certified examination review.
