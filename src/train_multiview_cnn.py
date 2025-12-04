#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for seizure-detector project.

- Reads combined multi-view npy files from PROJECT_ROOT/data/npy
- Trains CNN models for different view combinations:
    TOP only, SIDE only, SIDE+SIDE2, TOP+SIDE, TOP+SIDE+SIDE2
- Evaluates each model and saves:
    - model_*.h5 in PROJECT_ROOT/results
    - confusion_matrix_*.png in PROJECT_ROOT/results
    - metrics_summary.csv with all metrics

Author: Hyojin Seo
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ======================
# Configuration
# ======================

# Project paths (aligned with preprocessing.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NPY_DIR = PROJECT_ROOT / "data" / "npy"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DTYPE = np.float32

# Train/validation split
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Training
EPOCHS = 10
BATCH_SIZE = 32

# Threshold for converting probabilities -> binary labels
THRESHOLD = 0.5

# Combined view files (produced by preprocessing.py)
VIEW_PATHS = {
    "TOP": NPY_DIR / "X_TOP_combined.npy",
    "SIDE": NPY_DIR / "X_SIDE_combined.npy",
    "SIDE2": NPY_DIR / "X_SIDE2_combined.npy",
}

# View combinations to compare
VIEW_COMBINATIONS = {
    "TOP only": ["TOP"],
    "SIDE only": ["SIDE"],
    "SIDE+SIDE2": ["SIDE", "SIDE2"],
    "TOP+SIDE": ["TOP", "SIDE"],
    "TOP+SIDE+SIDE2": ["TOP", "SIDE", "SIDE2"],
}

# Global RNG (like preprocessing)
RNG = np.random.default_rng(RANDOM_SEED)

# ======================
# Sanity checks & label loading
# ======================

Y_PATH = NPY_DIR / "y_combined.npy"
if not Y_PATH.exists():
    raise FileNotFoundError(f"Missing labels file: {Y_PATH}")

y = np.load(Y_PATH).astype(np.uint8)
print(f"Using label file: {Y_PATH} | Shape: {y.shape} | Positives: {int(y.sum())}")


# ======================
# Helper functions
# ======================

def open_views_memmap(views):
    """
    Open requested views as memmaps and check shapes.

    Returns:
        memmaps: list of memmaps (one per view) with shape (N, H, W, 1)
        input_shape: (H, W, C) where C = number of views
        N: number of samples
    """
    memmaps = []
    N = None
    H = W = None
    C = 0

    for v in views:
        path = VIEW_PATHS.get(v)
        if path is None:
            raise KeyError(f"Unknown view key: {v}")
        if not path.exists():
            raise FileNotFoundError(f"Missing array for view {v}: {path}")

        arr = np.load(path, mmap_mode="r")

        # Expect (N, H, W, 1)
        if arr.ndim != 4 or arr.shape[-1] != 1:
            raise ValueError(f"Expected shape (N, H, W, 1) for {v}, got {arr.shape}")

        if N is None:
            N, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
        else:
            if arr.shape[0] != N or arr.shape[1] != H or arr.shape[2] != W:
                raise ValueError(
                    f"Shape mismatch across views. "
                    f"{v} has {arr.shape}, expected (N, {H}, {W}, 1)"
                )

        memmaps.append(arr)
        C += 1

    if len(y) != N:
        raise ValueError(f"y length {len(y)} != X length {N}")

    input_shape = (H, W, C)
    return memmaps, input_shape, N


def build_model(input_shape):
    """
    Build a simple CNN for binary seizure detection.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_index_splits(y_array, test_size=0.2, seed=42):
    """
    Return train_idx, test_idx arrays using stratified split.
    """
    return train_test_split(
        np.arange(len(y_array)),
        test_size=test_size,
        stratify=y_array,
        random_state=seed,
    )


def streaming_dataset(memmaps, idx, batch_size=32, shuffle=False, repeat=False):
    """
    Create a tf.data.Dataset that reads rows from memmaps on-the-fly and
    concatenates views on the channel axis.

    Assumes each memmap is (N, H, W, 1).
    """
    N = len(idx)
    H, W = memmaps[0].shape[1], memmaps[0].shape[2]
    C = len(memmaps)

    def gen():
        order = np.array(idx, dtype=np.int64)

        while True:
            if shuffle:
                RNG.shuffle(order)

            start = 0
            while start < N:
                end = min(start + batch_size, N)
                batch_idx = order[start:end]

                Xb = np.empty((len(batch_idx), H, W, C), dtype=DTYPE)
                for c, mm in enumerate(memmaps):
                    # Arrays already normalized to [0, 1] in preprocessing.
                    Xb[..., c] = mm[batch_idx, ..., 0]

                yb = y[batch_idx].astype(np.float32)
                yield Xb, yb

                start = end

            if not repeat:
                break

    output_signature = (
        tf.TensorSpec(shape=(None, H, W, C), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_model(model, memmaps, idx, batch_size=32, threshold=0.5):
    """
    Compute confusion matrix and metrics on given indices.
    """
    H, W = memmaps[0].shape[1], memmaps[0].shape[2]
    C = len(memmaps)

    y_true = y[idx].astype(int)
    preds = []

    start = 0
    while start < len(idx):
        end = min(start + batch_size, len(idx))
        batch_idx = idx[start:end]

        Xb = np.empty((len(batch_idx), H, W, C), dtype=DTYPE)
        for c, mm in enumerate(memmaps):
            Xb[..., c] = mm[batch_idx, ..., 0]

        pb = model.predict(Xb, verbose=0).ravel()
        preds.append(pb)

        start = end

    y_prob = np.concatenate(preds)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return cm, acc, precision, recall, f1


def save_confmat(cm, title, out_path: Path):
    """
    Save a confusion matrix plot to out_path.
    """
    plt.figure(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Seizure", "Seizure"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ======================
# Main training loop
# ======================

print("\nPreparing train/validation split...")
train_idx, test_idx = make_index_splits(
    y, test_size=TEST_SIZE, seed=RANDOM_SEED
)

device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
print(f"Device: {device}")
print(
    f"Train N={len(train_idx)} | Val N={len(test_idx)} | "
    f"Total Positives={int(y.sum())}"
)

all_results = []

for label, views in VIEW_COMBINATIONS.items():
    print(f"\n=== Training model: {label} ===")
    print(f"Views: {views}")

    # Open view arrays as memmaps
    memmaps, input_shape, N = open_views_memmap(views)
    print(f"Total samples: {N} | Input shape: {input_shape}")

    steps_per_epoch = math.ceil(len(train_idx) / BATCH_SIZE)

    # Datasets (streaming, aligned with preprocessing data layout)
    train_ds = streaming_dataset(
        memmaps,
        train_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
        repeat=True,
    )
    test_ds = streaming_dataset(
        memmaps,
        test_idx,
        batch_size=BATCH_SIZE,
        shuffle=False,
        repeat=False,
    )

    # Build & train model
    model = build_model(input_shape)
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds,
        verbose=1,
    )

    # Evaluate
    cm, acc, precision, recall, f1 = evaluate_model(
        model,
        memmaps,
        test_idx,
        batch_size=BATCH_SIZE,
        threshold=THRESHOLD,
    )

    # Save model
    model_filename = f"model_{label.replace('+', '_').replace(' ', '_')}.h5"
    model_path = RESULTS_DIR / model_filename
    model.save(model_path)

    # Save confusion matrix image
    cm_filename = f"confusion_matrix_{label.replace('+', '_').replace(' ', '_')}.png"
    cm_path = RESULTS_DIR / cm_filename
    save_confmat(cm, f"Confusion Matrix: {label}", cm_path)

    # Record metrics
    result_row = {
        "Model": label,
        "Accuracy": acc,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1 Score": f1,
        "True Positives": int(cm[1, 1]),
        "Total Positives": int(cm[1].sum()),
        "Model File": model_filename,
        "Confusion Matrix": cm_filename,
    }
    all_results.append(result_row)

    print(
        f"Acc={acc:.4f}  Prec={precision:.4f}  "
        f"Recall={recall:.4f}  F1={f1:.4f}"
    )
    print("Confusion Matrix:\n", cm)

# Save summary CSV
df = pd.DataFrame(all_results)
csv_path = RESULTS_DIR / "metrics_summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved all metrics to: {csv_path}")
print("\nTraining complete.")
