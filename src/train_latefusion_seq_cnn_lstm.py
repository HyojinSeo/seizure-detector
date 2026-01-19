#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train late-fusion TimeDistributed CNN + LSTM for seizure detection.

Input data_dir:
  X_TOP.npy / X_SIDE.npy / X_SIDE2.npy (depending on views)
  y.npy
  meta.json

Example:
  python -m src.train_latefusion_seq_cnn_lstm \
    --data_dir data/processed_seq/latefusion/top_side \
    --results_dir results/latefusion/top_side \
    --epochs 20 --batch_size 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd

class MemmapSequence(tf.keras.utils.Sequence):
    def __init__(self, X: Dict[str, np.ndarray], y: np.ndarray, views: List[str], idx: np.ndarray, batch_size: int, shuffle: bool = True):
        self.X = X
        self.y = y
        self.views = views
        self.idx = np.array(idx, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.idx) / self.batch_size))

    def __getitem__(self, i: int):
        sl = self.idx[i * self.batch_size : (i + 1) * self.batch_size]
        inputs = {v: np.array(self.X[v][sl], dtype=np.float32) for v in self.views}  # batch만 RAM으로
        labels = np.array(self.y[sl], dtype=np.int32).reshape(-1, 1)
        return inputs, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

def load_meta(data_dir: Path) -> Dict:
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_arrays(data_dir: Path, views: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    X = {v: np.load(data_dir / f"X_{v}.npy", mmap_mode="r") for v in views}
    y = np.load(data_dir / "y.npy", mmap_mode="r").astype(np.int32)
    return X, y


def split_indices(N: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(N * val_split))
    return idx[n_val:], idx[:n_val]


def make_ds(X: Dict[str, np.ndarray], y: np.ndarray, views: List[str], idx: np.ndarray, batch_size: int) -> tf.data.Dataset:
    inputs = {v: X[v][idx] for v in views}
    labels = y[idx]
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_branch(seq_len: int, H: int, W: int, C: int, name: str) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, H, W, C), name=f"{name}_in")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"))(inp)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    return tf.keras.Model(inp, x, name=f"{name}_branch")


def build_model(views: List[str], shapes: Dict[str, Tuple[int, int, int, int, int]], lr: float) -> tf.keras.Model:
    # Ensure all views have same (T,H,W,C)
    _, T, H, W, C = shapes[views[0]]
    inputs = {}
    feats = []

    for v in views:
        shp = shapes[v]
        if shp[1:] != (T, H, W, C):
            raise ValueError(f"Shape mismatch: {v} has {shp}, expected (*,{T},{H},{W},{C})")

        inp = tf.keras.Input(shape=(T, H, W, C), name=v)
        branch = build_branch(T, H, W, C, name=v)
        fv = branch(inp)  # (batch, T, F)
        inputs[v] = inp
        feats.append(fv)

    fused = feats[0] if len(feats) == 1 else tf.keras.layers.Concatenate(axis=-1, name="feat_concat")(feats)

    x = tf.keras.layers.LSTM(128, return_sequences=False)(fused)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="seizure_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out, name="latefusion_seq_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--results_dir", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cm_threshold", type=float, default=0.5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(data_dir)
    views = meta["views"]

    X, y = load_arrays(data_dir, views)
    N = int(y.shape[0])
    if N == 0:
        raise RuntimeError("Empty dataset.")

    shapes = {v: X[v].shape for v in views}

    tr_idx, va_idx = split_indices(N, args.val_split, args.seed)
    train_seq = MemmapSequence(X, y, views, tr_idx, args.batch_size, shuffle=True)
    val_seq   = MemmapSequence(X, y, views, va_idx, args.batch_size, shuffle=False)

    model = build_model(views, shapes, lr=args.lr)

    with open(results_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(results_dir / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump({"views": views, "shapes": {k: list(v) for k, v in shapes.items()}, "N": N}, f, indent=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(results_dir / "train_log.csv")),
    ]

    history = model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=1)

    model.save(str(results_dir / "final_model.keras"))
    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    eval_vals = model.evaluate(val_seq, verbose=0)
    metrics = {k: float(v) for k, v in zip(model.metrics_names, eval_vals)}
    with open(results_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
      
    best_path = results_dir / "best_model.keras"
    if best_path.exists():
      model = tf.keras.models.load_model(best_path)
    cm_root = Path.home() / "seizure-detector" / "results" / "confusion_matrices"
    cm_root.mkdir(parents=True, exist_ok=True)

    exp_name = results_dir.name  # e.g. top_side
    cm_dir = cm_root / exp_name
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = []
    y_pred = []
    for inputs, labels in val_seq:
        probs = model.predict(inputs, verbose=0)
        preds = (probs >= args.cm_threshold).astype(int)
        y_true.extend(labels.reshape(-1).tolist())
        y_pred.extend(preds.reshape(-1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
      cm,
      index=["true_0", "true_1"],
      columns=["pred_0", "pred_1"]
    )
    cm_path = cm_dir / f"confusion_matrix_val_thr{args.cm_threshold:.2f}.csv"
    cm_df.to_csv(cm_path, index=True)
    
    print(f"[OK] Saved {cm_path}")

    print("[OK] Training done.")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")


if __name__ == "__main__":
    main()
