"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         F R A C T U R E V I S I O N                         ║
║          Automated Forensic Glass Fracture Analysis System v1.0              ║
║──────────────────────────────────────────────────────────────────────────────║
║  Modules:  1. Preprocessing    2. Classification (ResNet-50)                 ║
║            3. Point of Impact  4. Evaluation                                 ║
║  Runtime:  Google Colab (GPU recommended)                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

FORENSIC CONTEXT
────────────────
When a projectile or object strikes glass, it produces two distinct fracture
families:
  • Radial fractures  – propagate OUTWARD from the impact point like spokes.
  • Concentric rings  – circle the impact point perpendicular to the radials.

High-Velocity impacts (gunshots, high-speed projectiles):
  - Dense, tightly spaced radial lines
  - Small, well-defined entry hole
  - Fracture lines stop at pre-existing cracks (Locard principle for glass)

Low-Velocity impacts (blunt objects, hammers, thrown rocks):
  - Sparse, irregular radial lines
  - Large, irregular fracture zone
  - Extensive concentric ring damage

This system automates that distinction and pinpoints the origin.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 ─ COLAB SETUP & DEPENDENCY INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────
# Run this cell first in Google Colab:
#
#   !pip install -q opencv-python-headless tensorflow scikit-learn matplotlib
#                    numpy pillow seaborn
#
# For GPU acceleration (recommended for ResNet-50 training):
#   Runtime -> Change runtime type -> GPU
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import warnings
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120

# ── Reproducibility seed ──────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Global constants ──────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)   # ResNet-50 native input size
NUM_CLASSES     = 2            # 0 = Low-Velocity, 1 = High-Velocity
CLASS_NAMES     = ["Low-Velocity", "High-Velocity"]
BATCH_SIZE      = 16
EPOCHS_WARMUP   = 5            # Train only the head (frozen backbone)
EPOCHS_FINETUNE = 15           # Unfreeze top ResNet layers and fine-tune
LEARNING_RATE   = 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
"""
FORENSIC RATIONALE
──────────────────
Raw crime-scene photographs contain lighting artefacts, blood, tape marks, and
scale rulers that confuse a classifier. The preprocessing pipeline isolates the
pure fracture "skeleton":

  1. Greyscale conversion strips colour information irrelevant to crack geometry.
  2. Gaussian Blur (kernel 5x5) suppresses photographic grain before edge
     detection; sigma chosen empirically for glass macro photography.
  3. Canny Edge Detection finds true fracture edges using a dual-threshold:
       low_thresh  = 50  -> includes faint concentric rings
       high_thresh = 150 -> anchors strong radial lines
  4. Morphological dilation (1 iteration, 3x3) bridges micro-gaps in
     fracture lines caused by specular reflections on glass surfaces.
"""

def load_and_validate_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk, validate it, and return a BGR numpy array.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray  BGR image, or None if the file cannot be read.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] File not found: {image_path}")
        return None

    img = cv2.imread(str(path))
    if img is None:
        print(f"[ERROR] OpenCV could not decode: {image_path}")
        return None

    print(f"[INFO] Loaded '{path.name}'  shape={img.shape}  dtype={img.dtype}")
    return img


def preprocess_fracture_image(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int   = 50,
    canny_high: int  = 150,
    dilate_iter: int = 1,
    debug: bool      = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a raw glass-fracture photograph into a clean fracture skeleton.

    Processing chain:
        BGR image -> greyscale -> Gaussian blur -> Canny edges -> dilation

    Parameters
    ----------
    image       : BGR numpy array from load_and_validate_image()
    blur_kernel : Side length of the Gaussian kernel (must be odd).
    canny_low   : Lower hysteresis threshold for Canny.
    canny_high  : Upper hysteresis threshold for Canny.
    dilate_iter : Number of morphological dilation passes.
    debug       : If True, displays every intermediate stage inline.

    Returns
    -------
    (gray, blurred, skeleton) tuple of numpy arrays.
    """
    # Stage 1: Greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stage 2: Gaussian Blur
    # Kernel must be odd; larger kernels smooth more aggressively.
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), sigmaX=0)

    # Stage 3: Canny Edge Detection
    # Pixels above canny_high -> definite edge (strong)
    # Pixels between low/high -> edge only if connected to a strong edge (weak)
    # Pixels below canny_low  -> suppressed
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Stage 4: Morphological Dilation
    kernel   = np.ones((3, 3), np.uint8)
    skeleton = cv2.dilate(edges, kernel, iterations=dilate_iter)

    if debug:
        _show_preprocessing_stages(image, gray, blurred, edges, skeleton)

    return gray, blurred, skeleton


def _show_preprocessing_stages(
    original: np.ndarray,
    gray: np.ndarray,
    blurred: np.ndarray,
    edges: np.ndarray,
    skeleton: np.ndarray
) -> None:
    """Inline debug visualisation for preprocessing pipeline."""
    titles = ["Original (BGR)", "Greyscale", "Gaussian Blur",
              "Canny Edges", "Dilated Skeleton"]
    images = [
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        gray, blurred, edges, skeleton
    ]
    cmaps = [None, "gray", "gray", "gray", "gray"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Preprocessing Pipeline — Fracture Skeleton Extraction",
                 fontsize=13, fontweight="bold", color="#1a1a2e")

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def batch_preprocess(image_dir: str, output_dir: str = None) -> List[Dict]:
    """
    Preprocess an entire directory of fracture images.

    Parameters
    ----------
    image_dir  : Directory containing raw .jpg / .png fracture images.
    output_dir : If provided, saves skeleton images here for archival.

    Returns
    -------
    List of dicts with keys: 'path', 'gray', 'blurred', 'skeleton'
    """
    results = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in Path(image_dir).iterdir()
             if p.suffix.lower() in extensions]

    if not paths:
        print(f"[WARN] No images found in '{image_dir}'")
        return results

    print(f"[INFO] Processing {len(paths)} images from '{image_dir}' ...")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for p in paths:
        img = load_and_validate_image(str(p))
        if img is None:
            continue
        gray, blurred, skeleton = preprocess_fracture_image(img)
        entry = {
            "path": str(p), "gray": gray,
            "blurred": blurred, "skeleton": skeleton
        }
        results.append(entry)

        if output_dir:
            out_path = Path(output_dir) / f"skeleton_{p.stem}.png"
            cv2.imwrite(str(out_path), skeleton)

    print(f"[INFO] Preprocessing complete. {len(results)} images ready.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CLASSIFICATION (ResNet-50 Transfer Learning)
# ══════════════════════════════════════════════════════════════════════════════
"""
FORENSIC RATIONALE
──────────────────
Fracture density signatures are visually subtle; hand-crafted features (e.g.
line count alone) are unreliable across different glass thicknesses, camera
distances, and lighting. A convolutional backbone pre-trained on ImageNet
already encodes low-level texture detectors (edges, curves, grids) in its
early layers — exactly what fracture patterns consist of.

ResNet-50 was selected because:
  - Its residual connections prevent gradient vanishing during fine-tuning.
  - 50-layer depth provides sufficient capacity for texture discrimination.
  - Pre-trained weights are publicly available via Keras Applications.

Training strategy (two-phase):
  Phase 1 – Feature Extraction (head warm-up):
      Backbone frozen; only the custom classification head trains.
      Prevents early destruction of ImageNet features.
  Phase 2 – Fine-tuning (top layers unfrozen):
      Top 30 ResNet layers + head retrain at 10x lower LR.
      Adapts high-level features to fracture-specific patterns.

Data Augmentation:
  Crime-scene photographs are taken at inconsistent angles, distances, and
  orientations. Augmentation artificially diversifies training data:
    - rotation_range=30deg  -> compensates for camera tilt
    - zoom_range=0.2        -> compensates for varying distances
    - horizontal/vert flip  -> glass fractures have no canonical orientation
    - brightness_range      -> compensates for flash / ambient light differences
"""

def build_resnet50_classifier(
    num_classes: int     = NUM_CLASSES,
    img_size: tuple      = IMG_SIZE,
    learning_rate: float = LEARNING_RATE
) -> Model:
    """
    Build a binary classifier on top of a frozen ResNet-50 backbone.

    Architecture
    ────────────
    Input (224x224x3)
        |
    ResNet-50 backbone (frozen, ImageNet weights)
        |
    GlobalAveragePooling2D      <- collapses spatial dims, keeps channels
        |
    Dense(256, relu)            <- task-specific feature combination
        |
    BatchNormalization          <- stabilises training
        |
    Dropout(0.5)                <- regularisation
        |
    Dense(128, relu)
        |
    Dropout(0.3)
        |
    Dense(num_classes, softmax) <- probability over [Low-Vel, High-Vel]

    Parameters
    ----------
    num_classes    : Number of output classes (default 2).
    img_size       : (height, width) tuple; must match preprocessing resize.
    learning_rate  : Adam optimiser initial learning rate.

    Returns
    -------
    Compiled Keras Model ready for phase-1 training.
    """
    input_tensor = layers.Input(shape=(*img_size, 3), name="fracture_input")

    # Backbone (frozen)
    backbone = ResNet50(
        weights      = "imagenet",
        include_top  = False,        # remove ImageNet classification head
        input_tensor = input_tensor
    )
    backbone.trainable = False       # freeze all backbone weights for Phase 1

    # Custom Classification Head
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(0.5, name="drop_256")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="drop_128")(x)
    output = layers.Dense(
        num_classes, activation="softmax", name="impact_class"
    )(x)

    model = Model(inputs=input_tensor, outputs=output,
                  name="FractureVision_ResNet50")

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    print(f"\n{'─'*60}")
    print("FractureVision Model Summary")
    print(f"{'─'*60}")
    model.summary()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Trainable params (head only): {trainable_params:,}")
    return model


def unfreeze_top_layers(
    model: Model,
    n_layers: int  = 30,
    new_lr: float  = 1e-5
) -> Model:
    """
    Phase 2: Unfreeze the top n_layers of the ResNet backbone for fine-tuning.
    Modified to handle flat model architectures.
    """
    try:
        backbone = model.get_layer("resnet50")
    except ValueError:
        backbone = model

    backbone.trainable = True

    layers_to_toggle = []
    for layer in backbone.layers:
        layers_to_toggle.append(layer)
        if layer.name == "gap":
            break

    for layer in layers_to_toggle[:-n_layers]:
        layer.trainable = False
    
    for layer in layers_to_toggle[-n_layers:]:
        layer.trainable = True

    print(f"[INFO] Fine-tuning prepared: Top {n_layers} backbone layers unfrozen.")

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=new_lr),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )
    return model
    """
    Phase 2: Unfreeze the top n_layers of the ResNet backbone for fine-tuning.

    Parameters
    ----------
    model    : Model from build_resnet50_classifier().
    n_layers : Number of layers to unfreeze from the end of the backbone.
    new_lr   : Reduced learning rate; prevents over-writing ImageNet weights.

    Returns
    -------
    Recompiled Model with unfrozen top layers.
    """
    backbone = model.get_layer("resnet50")
    backbone.trainable = True

    # Freeze all layers except the last n_layers
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    print(f"[INFO] Unfrozen {unfrozen} backbone layers for fine-tuning "
          f"(LR={new_lr}).")

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=new_lr),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )
    return model


def create_data_generators(
    train_dir: str,
    val_dir: str,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE
):
    """
    Build augmented training and validation data generators.

    Augmentation strategy (training only):
        - rotation_range = 30deg  — camera tilt compensation
        - width/height_shift      — partial framing adjustment
        - zoom_range = 0.2        — distance compensation
        - horizontal_flip         — fractures have no canonical orientation
        - vertical_flip           — same reason
        - brightness_range        — flash / ambient light variation
        - fill_mode = 'reflect'   — mirrors edge pixels to avoid black borders
                                    that would create false Canny detections

    Parameters
    ----------
    train_dir  : Path to training set root (sub-folders = class names).
    val_dir    : Path to validation set root.
    img_size   : Target (H, W) for resizing.
    batch_size : Images per mini-batch.

    Returns
    -------
    (train_gen, val_gen) Keras DirectoryIterator pair.
    """
    train_datagen = ImageDataGenerator(
        rescale           = 1.0 / 255,
        rotation_range    = 30,
        width_shift_range = 0.1,
        height_shift_range= 0.1,
        zoom_range        = 0.2,
        horizontal_flip   = True,
        vertical_flip     = True,
        brightness_range  = [0.7, 1.3],
        fill_mode         = "reflect"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)  # no augmentation

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="sparse", seed=SEED, shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode="sparse", seed=SEED, shuffle=False
    )

    print(f"\n[INFO] Training   samples : {train_gen.samples}")
    print(f"[INFO] Validation samples : {val_gen.samples}")
    print(f"[INFO] Class map          : {train_gen.class_indices}")
    return train_gen, val_gen


def train_model(
    model: Model,
    train_gen,
    val_gen,
    epochs_warmup: int   = EPOCHS_WARMUP,
    epochs_finetune: int = EPOCHS_FINETUNE,
    save_path: str       = "fracturevision_best.h5"
) -> Tuple[Model, dict]:
    """
    Execute the two-phase training protocol.

    Phase 1 (Warm-up)   : trains only the custom head (backbone frozen).
    Phase 2 (Fine-tune) : unfreezes top 30 ResNet layers, continues training.

    Callbacks
    ---------
    EarlyStopping     — stops if val_loss stagnates for 5 epochs
    ReduceLROnPlateau — halves LR if val_loss plateaus for 3 epochs
    ModelCheckpoint   — saves the best val_accuracy weights

    Parameters
    ----------
    model          : Compiled Keras Model from build_resnet50_classifier().
    train_gen      : Training data generator.
    val_gen        : Validation data generator.
    epochs_warmup  : Epochs for Phase 1.
    epochs_finetune: Epochs for Phase 2.
    save_path      : File path to save best model weights.

    Returns
    -------
    (trained_model, combined_history_dict)
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, verbose=1, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(
            save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        )
    ]

    # Phase 1: Head Warm-up
    print(f"\n{'='*60}")
    print(" PHASE 1 — Feature Extraction (backbone frozen)")
    print(f"{'='*60}")
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs_warmup, callbacks=callbacks, verbose=1
    )

    # Phase 2: Fine-tuning
    print(f"\n{'='*60}")
    print(" PHASE 2 — Fine-tuning (top 30 ResNet layers unfrozen)")
    print(f"{'='*60}")
    model = unfreeze_top_layers(model, n_layers=30, new_lr=1e-5)
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs_finetune, callbacks=callbacks,
        initial_epoch=epochs_warmup, verbose=1
    )

    # Merge histories
    history = {}
    for key in h1.history:
        history[key] = h1.history[key] + h2.history.get(key, [])

    print(f"\n[INFO] Best model saved -> '{save_path}'")
    plot_training_history(history)
    return model, history


def plot_training_history(history: dict) -> None:
    """Plot accuracy and loss curves with Phase 1/2 boundary marker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FractureVision — Training History",
                 fontsize=14, fontweight="bold")

    epochs = range(1, len(history["accuracy"]) + 1)

    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"]
    ):
        ax.plot(epochs, history[metric[0]], "b-o", ms=4,
                label=f"Train {title}")
        ax.plot(epochs, history[metric[1]], "r-o", ms=4,
                label=f"Val {title}")
        ax.axvline(x=EPOCHS_WARMUP + 0.5, color="green",
                   linestyle="--", linewidth=1.5,
                   label="Fine-tune start")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — POINT OF IMPACT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
"""
FORENSIC RATIONALE
──────────────────
The Point of Impact (POI) is the most critical measurement in glass fracture
analysis — it identifies exactly where the projectile or object struck.

Algorithm:
  1. Hough Line Transform detects straight radial fracture lines in the
     preprocessed skeleton image. The probabilistic variant (HoughLinesP)
     is preferred because it:
       - Tolerates fragmented lines (real fractures are not perfectly straight)
       - Returns finite line segments (easier to extend toward convergence)
       - Runs faster on high-resolution crime-scene images

  2. Detected lines are converted to the parametric form ax + by = c
     (normal-vector form; avoids division-by-zero at vertical lines).

  3. Convergence estimated by LEAST SQUARES intersection of all lines:
       A * p = c   where p = [x, y]
       Solution:   p* = (A^T A)^-1 A^T c
     More robust than pairwise averaging, which amplifies noise from
     near-parallel lines.

  4. A confidence scatter of pairwise intersections shows the forensic
     uncertainty envelope around the POI estimate.
"""

def detect_radial_lines(
    skeleton: np.ndarray,
    rho: float        = 1,
    theta: float      = np.pi / 180,
    threshold: int    = 50,
    min_line_len: int = 40,
    max_line_gap: int = 10
) -> Optional[np.ndarray]:
    """
    Apply Probabilistic Hough Line Transform to detect radial fracture lines.

    Parameters
    ----------
    skeleton      : Binary edge image from preprocess_fracture_image().
    rho           : Distance resolution of the accumulator in pixels.
    theta         : Angle resolution in radians (1-degree default).
    threshold     : Minimum accumulator votes for a line to be accepted.
                    Lower -> more lines detected (risk: noise);
                    Higher -> fewer, more confident lines.
    min_line_len  : Minimum pixel length of a segment to be retained.
                    Set >= 40px to exclude short artefacts (e.g. dust).
    max_line_gap  : Max gap in pixels between collinear segments to merge.

    Returns
    -------
    Array of shape (N, 1, 4) with [x1, y1, x2, y2] per line, or None.
    """
    lines = cv2.HoughLinesP(
        skeleton,
        rho           = rho,
        theta         = theta,
        threshold     = threshold,
        minLineLength = min_line_len,
        maxLineGap    = max_line_gap
    )

    if lines is None:
        print("[WARN] No radial lines detected. "
              "Try lowering 'threshold' or 'min_line_len'.")
        return None

    print(f"[INFO] Detected {len(lines)} radial line segments.")
    return lines


def compute_convergence_point(
    lines: np.ndarray
) -> Tuple[Optional[Tuple[float, float]], Optional[np.ndarray]]:
    """
    Compute the least-squares convergence point of detected radial lines.

    Each segment (x1,y1)->(x2,y2) is converted to normal-vector form:
        a_i * x + b_i * y = c_i
    where [a_i, b_i] is the unit normal to the line direction.

    The over-determined system A*p = c is solved via pseudo-inverse:
        p* = (A^T A)^-1 A^T c

    Parameters
    ----------
    lines : Hough line segments array from detect_radial_lines().

    Returns
    -------
    (convergence_point, all_pairwise_intersections)
      convergence_point : (x, y) float tuple — estimated POI.
      intersections     : numpy array of pairwise intersection points
                          used for uncertainty visualisation.
    """
    if lines is None or len(lines) < 2:
        print("[ERROR] Need at least 2 lines to compute convergence.")
        return None, None

    A_rows, c_vals, intersections = [], [], []

    # Build the normal-form linear system
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        # Unit normal perpendicular to line direction
        a, b = -dy / length, dx / length
        c = a * x1 + b * y1
        A_rows.append([a, b])
        c_vals.append(c)

    if len(A_rows) < 2:
        print("[ERROR] Insufficient valid line equations.")
        return None, None

    A = np.array(A_rows)
    c = np.array(c_vals)

    # Least-squares solution via pseudo-inverse
    try:
        ATA = A.T @ A
        ATc = A.T @ c
        point = np.linalg.solve(ATA, ATc)
    except np.linalg.LinAlgError:
        print("[ERROR] Singular matrix — lines may be nearly parallel.")
        return None, None

    # Pairwise intersections for uncertainty scatter
    for i in range(len(A_rows)):
        for j in range(i + 1, len(A_rows)):
            M   = np.array([A_rows[i], A_rows[j]])
            rhs = np.array([c_vals[i], c_vals[j]])
            if abs(np.linalg.det(M)) > 1e-6:
                pt = np.linalg.solve(M, rhs)
                intersections.append(pt)

    ints_array = np.array(intersections) if intersections else None
    print(f"[INFO] Estimated POI: x={point[0]:.1f}, y={point[1]:.1f}  "
          f"({len(intersections)} pairwise intersections)")
    return (float(point[0]), float(point[1])), ints_array


def visualize_fracture_analysis(
    original_image: np.ndarray,
    skeleton: np.ndarray,
    lines: Optional[np.ndarray],
    convergence_point: Optional[Tuple[float, float]],
    intersections: Optional[np.ndarray],
    impact_class: str  = "Unknown",
    confidence: float  = 0.0,
    save_path: Optional[str] = None
) -> None:
    """
    Create a forensic-grade composite visualisation with four panels.

    Panel A — Original photograph
    Panel B — Preprocessed fracture skeleton
    Panel C — Detected radial Hough lines
    Panel D — Final annotated overlay with POI marker

    Colour coding (forensic convention):
        Green   -> detected radial fracture lines
        Red X   -> estimated Point of Impact (high-velocity)
        Blue X  -> estimated Point of Impact (low-velocity)
        Yellow  -> pairwise intersection uncertainty scatter
        Cyan    -> approximate concentric fracture zones

    Parameters
    ----------
    original_image    : Raw BGR image.
    skeleton          : Binary skeleton from preprocessing.
    lines             : Hough line segments.
    convergence_point : (x, y) POI estimate.
    intersections     : Pairwise intersection points.
    impact_class      : "High-Velocity" or "Low-Velocity".
    confidence        : Model softmax confidence [0, 1].
    save_path         : Optional path to save the figure.
    """
    orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    h, w     = original_image.shape[:2]

    # Panel C: draw green lines on skeleton background
    lines_overlay = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Panel D: POI annotation overlay on original image
    annotated  = orig_rgb.copy()
    poi_color  = (255, 50, 50) if "High" in impact_class else (50, 50, 255)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            _draw_extended_line(annotated, x1, y1, x2, y2,
                                color=(0, 255, 0), thickness=1, w=w, h=h)

    if convergence_point is not None:
        cx, cy = int(convergence_point[0]), int(convergence_point[1])

        # Uncertainty scatter
        if intersections is not None and len(intersections) > 2:
            for pt in intersections:
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(annotated, (px, py), 2, (255, 220, 0), -1)

        # Approximate concentric fracture zones
        cv2.circle(annotated, (cx, cy), 60, (0, 220, 255), 1)
        cv2.circle(annotated, (cx, cy), 30, (0, 220, 255), 1)

        # POI crosshair marker
        cv2.drawMarker(annotated, (cx, cy), poi_color,
                       cv2.MARKER_TILTED_CROSS, 30, 3)
        cv2.circle(annotated, (cx, cy), 8, poi_color, 2)

        # Coordinate label
        font_scale = max(0.5, min(w, h) / 800)
        cv2.putText(annotated, f"POI ({cx}, {cy})",
                    (cx + 12, cy - 12),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, poi_color, 2)

    # Figure layout
    fig = plt.figure(figsize=(18, 9), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.35, wspace=0.08,
        left=0.03, right=0.97, top=0.88, bottom=0.05
    )

    panel_A = fig.add_subplot(gs[:, 0:2])   # Original — large left panel
    panel_B = fig.add_subplot(gs[0, 2])
    panel_C = fig.add_subplot(gs[0, 3])
    panel_D = fig.add_subplot(gs[1, 2:4])   # Annotated — large bottom-right

    panels = [
        (panel_A, orig_rgb, "A — Original Photograph"),
        (panel_B, skeleton, "B — Fracture Skeleton"),
        (panel_C, cv2.cvtColor(lines_overlay, cv2.COLOR_BGR2RGB),
                             "C — Hough Radial Lines"),
        (panel_D, annotated, "D — Point of Impact Overlay"),
    ]

    for ax, img, title in panels:
        cmap = "inferno" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap, aspect="auto")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # Classification banner
    banner_color = "#8b0000" if "High" in impact_class else "#003580"
    if convergence_point:
        banner_text = (
            f"  CLASSIFICATION: {impact_class.upper()}  "
            f"|  CONFIDENCE: {confidence*100:.1f}%  "
            f"|  POINT OF IMPACT: "
            f"({int(convergence_point[0])}, {int(convergence_point[1])}) px"
        )
    else:
        banner_text = (
            f"  CLASSIFICATION: {impact_class.upper()}  "
            f"|  CONFIDENCE: {confidence*100:.1f}%  "
            f"|  POI: UNDETECTED"
        )

    fig.text(0.5, 0.94, banner_text,
             ha="center", va="center",
             fontsize=11, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=banner_color,
                       edgecolor="white", alpha=0.95))

    fig.text(
        0.5, 0.005,
        "FractureVision v1.0  |  Automated Forensic Glass Analysis  "
        "|  NOT a substitute for certified forensic examination",
        ha="center", va="bottom", fontsize=7,
        color="#888888", style="italic"
    )

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"[INFO] Visualisation saved -> '{save_path}'")

    plt.show()


def _draw_extended_line(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple, thickness: int, w: int, h: int
) -> None:
    """Extend a finite Hough segment to the full image boundary."""
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return
    scale = max(w, h) * 2
    ex1 = int(x1 - scale * dx)
    ey1 = int(y1 - scale * dy)
    ex2 = int(x2 + scale * dx)
    ey2 = int(y2 + scale * dy)
    cv2.line(img, (ex1, ey1), (ex2, ey2), color, thickness,
             lineType=cv2.LINE_AA)


def run_full_pipeline(
    image_path: str,
    model: Optional[Model] = None,
    save_output: Optional[str] = None
) -> Dict:
    """
    End-to-end analysis of a single glass fracture image.

    Steps:
        1. Load and validate the image.
        2. Preprocess to fracture skeleton.
        3. Classify impact type (if model is provided).
        4. Detect radial lines and estimate POI.
        5. Generate forensic visualisation.

    Parameters
    ----------
    image_path  : Path to the input image.
    model       : Trained Keras model (optional; skips classification if None).
    save_output : Path to save the output figure (optional).

    Returns
    -------
    Dict with keys: 'image', 'skeleton', 'lines', 'poi', 'class', 'confidence'
    """
    print(f"\n{'='*60}")
    print(f"  FractureVision Analysis: {Path(image_path).name}")
    print(f"{'='*60}")

    image = load_and_validate_image(image_path)
    if image is None:
        return {}

    _, _, skeleton = preprocess_fracture_image(image)

    impact_class, confidence = "Unknown", 0.0
    if model is not None:
        impact_class, confidence = classify_fracture(image, model)

    lines = detect_radial_lines(skeleton)
    poi, intersections = compute_convergence_point(lines)

    visualize_fracture_analysis(
        original_image    = image,
        skeleton          = skeleton,
        lines             = lines,
        convergence_point = poi,
        intersections     = intersections,
        impact_class      = impact_class,
        confidence        = confidence,
        save_path         = save_output
    )

    return {
        "image"     : image,
        "skeleton"  : skeleton,
        "lines"     : lines,
        "poi"       : poi,
        "class"     : impact_class,
        "confidence": confidence
    }


def classify_fracture(
    image: np.ndarray,
    model: Model
) -> Tuple[str, float]:
    """
    Run inference on a single BGR image using the trained ResNet-50 model.

    Parameters
    ----------
    image : BGR numpy array.
    model : Trained Keras Model.

    Returns
    -------
    (class_name, confidence) tuple.
    """
    img_resized = cv2.resize(image, IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm    = img_rgb.astype("float32") / 255.0
    img_batch   = np.expand_dims(img_norm, axis=0)  # shape (1, 224, 224, 3)

    probs = model.predict(img_batch, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
"""
FORENSIC RATIONALE
──────────────────
In forensic science, a tool's validity must be demonstrated on BLIND samples —
evidence the model has never seen during development. This mirrors the legal
standard for scientific evidence (Daubert / Frye criteria).

Metrics reported:
  Accuracy              — overall correct classification rate
  Precision             — P(true High-Vel | predicted High-Vel)
                          critical for avoiding false-positive bullet claims
  Recall (Sensitivity)  — P(predict High-Vel | true High-Vel)
                          critical for not missing bullet evidence
  F1-Score              — harmonic mean of Precision and Recall
  AUC-ROC               — discrimination ability across all thresholds
  Confusion Matrix      — breakdown of TP / TN / FP / FN
"""

def evaluate_model(
    model: Model,
    test_dir: str,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Evaluate the trained model on a held-out blind test set.

    Parameters
    ----------
    model      : Trained Keras Model.
    test_dir   : Path to test set root (sub-folders = class names).
    batch_size : Inference batch size.

    Returns
    -------
    Dict with keys: 'accuracy', 'auc', 'report', 'confusion_matrix',
                    'y_true', 'y_pred', 'y_proba'
    """
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=batch_size,
        class_mode="sparse", shuffle=False
    )

    y_true  = test_gen.classes
    y_proba = model.predict(test_gen, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auc      = (roc_auc_score(y_true, y_proba[:, 1])
                if NUM_CLASSES == 2 else 0.0)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred,
                                     target_names=CLASS_NAMES)

    print(f"\n{'='*60}")
    print("  EVALUATION REPORT — BLIND TEST SET")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"\n{report}")

    _plot_confusion_matrix(cm)

    return {
        "accuracy"        : accuracy,
        "auc"             : auc,
        "report"          : report,
        "confusion_matrix": cm,
        "y_true"          : y_true,
        "y_pred"          : y_pred,
        "y_proba"         : y_proba
    }


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """
    Render a publication-quality annotated confusion matrix.

    Left panel  — raw counts.
    Right panel — row-normalised rates (percentages).

    Color scale: deep red = diagonal (correct); deep blue = off-diagonal (errors).
    """
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FractureVision — Confusion Matrix (Blind Test Set)",
                 fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (Row %)"],
        ["d", ".2%"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            cmap="RdBu_r", linewidths=0.5,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0, vmax=(None if fmt == "d" else 1),
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()


def evaluate_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    dataset_name: str = "Test"
) -> Dict:
    """
    Evaluate pre-computed predictions (useful when running from saved arrays).

    Parameters
    ----------
    y_true       : Ground-truth integer labels.
    y_pred       : Predicted integer labels.
    y_proba      : Softmax probability array [N, num_classes] (optional).
    dataset_name : Label used in print outputs.

    Returns
    -------
    Dict with evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    auc = 0.0
    if y_proba is not None and NUM_CLASSES == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])

    print(f"\n{'='*60}")
    print(f"  EVALUATION — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy * 100:.2f}%   |   AUC-ROC: {auc:.4f}")
    print(f"\n{report}")
    _plot_confusion_matrix(cm)

    return {"accuracy": accuracy, "auc": auc,
            "report": report, "confusion_matrix": cm}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — SYNTHETIC DATA GENERATOR (Colab demo without real images)
# ══════════════════════════════════════════════════════════════════════════════
"""
NOTE: Real-world deployment requires labelled glass fracture photographs.
      This module generates physically-motivated synthetic images to enable
      end-to-end pipeline testing until a real dataset is acquired.

      Image structure mimics true fracture physics:
        High-Velocity -> 12-20 radial lines, dense concentric rings (3-6)
        Low-Velocity  -> 4-8 radial lines,  sparse concentric rings (1-3)
"""

def _draw_fracture_pattern(
    canvas: np.ndarray,
    cx: int, cy: int,
    n_radial: int,
    n_concentric: int,
    line_thickness: int,
    irregularity: float = 0.15
) -> np.ndarray:
    """Draw radial + concentric fractures on a canvas around (cx, cy)."""
    h, w = canvas.shape[:2]

    # Radial lines — evenly spaced with random angular jitter
    for i in range(n_radial):
        angle = (2 * math.pi * i / n_radial) + \
                np.random.uniform(-irregularity, irregularity)
        r = max(w, h) * 1.5
        ex = int(cx + r * math.cos(angle))
        ey = int(cy + r * math.sin(angle))
        cv2.line(canvas, (cx, cy), (ex, ey),
                 (200, 200, 200), line_thickness)

    # Concentric arcs — circles with slight random radius variation
    for k in range(1, n_concentric + 1):
        radius = int(k * min(h, w) / (2 * (n_concentric + 1)))
        radius += np.random.randint(-8, 8)
        cv2.circle(canvas, (cx, cy), max(radius, 5),
                   (180, 180, 180), max(1, line_thickness - 1))

    return canvas


def generate_synthetic_dataset(
    output_dir: str  = "/tmp/fracture_dataset",
    n_per_class: int = 50
) -> str:
    """
    Generate a labelled synthetic fracture dataset for pipeline testing.

    Directory structure created:
        output_dir/
            train/
                high_velocity/   img_0000.png ...
                low_velocity/    img_0000.png ...
            val/
                high_velocity/
                low_velocity/
            test/
                high_velocity/
                low_velocity/

    Split ratios: 70% train, 15% val, 15% test.

    Parameters
    ----------
    output_dir  : Root directory for the dataset.
    n_per_class : Total synthetic images per class before splitting.

    Returns
    -------
    output_dir path string.
    """
    for split in ["train", "val", "test"]:
        for cls in ["high_velocity", "low_velocity"]:
            Path(f"{output_dir}/{split}/{cls}").mkdir(
                parents=True, exist_ok=True)

    # Physics-inspired parameters
    config = {
        "high_velocity": dict(
            n_radial_range=(12, 20), n_conc_range=(3, 6),
            thickness=1, noise_std=10
        ),
        "low_velocity": dict(
            n_radial_range=(4, 8),  n_conc_range=(1, 3),
            thickness=2, noise_std=30
        )
    }

    splits     = {"train": 0.70, "val": 0.15, "test": 0.15}
    total      = n_per_class
    img_count  = 0

    print(f"[INFO] Generating synthetic dataset -> '{output_dir}'")

    for cls, cfg in config.items():
        indices = np.arange(total)
        np.random.shuffle(indices)
        boundaries = [
            0,
            int(total * splits["train"]),
            int(total * (splits["train"] + splits["val"])),
            total
        ]

        for s_idx, split in enumerate(list(splits.keys())):
            idxs = indices[boundaries[s_idx]: boundaries[s_idx + 1]]

            for img_idx in idxs:
                h, w = 400, 400
                # Dark background simulating backlit glass photography
                canvas = np.full((h, w, 3), 30, dtype=np.uint8)
                canvas += np.random.randint(
                    0, cfg["noise_std"], (h, w, 3), dtype=np.uint8
                )

                # POI offset from center to simulate off-centre impacts
                cx = w // 2 + np.random.randint(-30, 30)
                cy = h // 2 + np.random.randint(-30, 30)

                n_rad  = np.random.randint(*cfg["n_radial_range"])
                n_conc = np.random.randint(*cfg["n_conc_range"])

                canvas = _draw_fracture_pattern(
                    canvas, cx, cy, n_rad, n_conc, cfg["thickness"]
                )

                # Random exposure variation (flash / ambient differences)
                alpha  = np.random.uniform(0.7, 1.3)
                canvas = np.clip(canvas * alpha, 0, 255).astype(np.uint8)

                fname = (f"{output_dir}/{split}/{cls}/"
                         f"img_{img_idx:04d}.png")
                cv2.imwrite(fname, canvas)
                img_count += 1

    print(f"[INFO] Generated {img_count} synthetic images "
          f"across train / val / test splits.")
    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — END-TO-END COLAB DEMO
# ══════════════════════════════════════════════════════════════════════════════

def main_colab_demo() -> None:
    """
    Complete end-to-end FractureVision demo on synthetic data.

    Recommended Colab execution:
        1. Upload this file to Colab or paste in cells.
        2. Run all cells (Runtime -> Run all).
        3. Call main_colab_demo() in the final cell.
        4. Replace synthetic dataset path with real labelled forensic images.

    For real deployment:
        Organise evidence photos as:
            /dataset/train/high_velocity/*.jpg
            /dataset/train/low_velocity/*.jpg
            /dataset/val/...
            /dataset/test/...
        Then call create_data_generators(), train_model(), evaluate_model().
    """
    print("\n" + "=" * 60)
    print("  FRACTUREVISION  —  End-to-End Colab Demo")
    print("=" * 60 + "\n")

    # 1. Generate synthetic data
    dataset_dir = generate_synthetic_dataset(
        output_dir  = "/tmp/fracture_dataset",
        n_per_class = 80
    )
    TRAIN_DIR = f"{dataset_dir}/train"
    VAL_DIR   = f"{dataset_dir}/val"
    TEST_DIR  = f"{dataset_dir}/test"

    # 2. Build ResNet-50 classifier
    model = build_resnet50_classifier(
        num_classes   = NUM_CLASSES,
        img_size      = IMG_SIZE,
        learning_rate = LEARNING_RATE
    )

    # 3. Create data generators
    train_gen, val_gen = create_data_generators(
        TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE
    )

    # 4. Train (two-phase)
    model, history = train_model(
        model, train_gen, val_gen,
        epochs_warmup   = EPOCHS_WARMUP,
        epochs_finetune = EPOCHS_FINETUNE,
        save_path       = "/tmp/fracturevision_best.h5"
    )

    # 5. Evaluate on blind test set
    eval_results = evaluate_model(model, TEST_DIR, BATCH_SIZE)

    # 6. Single-image POI demo (high-velocity synthetic)
    demo_canvas = np.full((400, 400, 3), 25, dtype=np.uint8)
    demo_canvas += np.random.randint(0, 15, (400, 400, 3), dtype=np.uint8)
    demo_canvas = _draw_fracture_pattern(
        demo_canvas, cx=200, cy=200,
        n_radial=16, n_concentric=4, line_thickness=1
    )
    demo_path = "/tmp/demo_hv_fracture.png"
    cv2.imwrite(demo_path, demo_canvas)

    results = run_full_pipeline(
        image_path  = demo_path,
        model       = model,
        save_output = "/tmp/fracturevision_output.png"
    )

    # Summary printout
    print("\n" + "-" * 60)
    print("  ANALYSIS SUMMARY")
    print("-" * 60)
    print(f"  Classification : {results.get('class', 'N/A')}")
    print(f"  Confidence     : {results.get('confidence', 0) * 100:.1f}%")
    poi = results.get("poi")
    if poi:
        print(f"  Point of Impact: ({poi[0]:.1f}, {poi[1]:.1f}) px")
    print(f"  Test Accuracy  : {eval_results['accuracy'] * 100:.2f}%")
    print(f"  AUC-ROC        : {eval_results['auc']:.4f}")
    print("-" * 60)
    print("\n[INFO] Demo complete.")
    print("[INFO] Replace '/tmp/fracture_dataset' with real forensic imagery")
    print("[INFO] to train on actual glass fracture evidence.\n")


# Entry point
if __name__ == "__main__":
    main_colab_demo()
