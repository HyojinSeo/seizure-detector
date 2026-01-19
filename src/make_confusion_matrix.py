#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute and save validation confusion matrix from an existing trained model.

Requires:
  data_dir/
    - meta.json
    - y.npy
    - X_TOP.npy / X_SIDE.npy / X_SIDE2.npy (depending on meta["views"])

Example:
  python -m src.make_confusion_matrix \
    --data_dir data/processed_seq/latefusion/top_side \
    --model_path results/latefusion/top_side/best_model.keras \
    --cm_threshold 0.30 \
    --val_split 0.2 \
    --seed 42 \
    --out_dir ~/seizure-detector/results/confusion_matrices
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix


class MemmapSequence(tf.keras.utils.Sequence):
    def __init__(self, X: Dict[str, np.ndarray], y: np.ndarray, views: List[str], idx: np.ndarray, batch_size: int):
        self.X = X
        self.y = y
        self.views = views
        self.idx = np.array(idx, dtype=np.int64)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return int(np.ceil(len(self.idx) / self.batch_size))

    def __getitem__(self, i: int):
        sl = self.idx[i * self.batch_size : (i + 1) * self.batch_size]
        inputs = {v: np.array(self.X[v][sl], dtype=np.float32) for v in self.views}
        labels = np.array(self.y[sl], dtype=np.int32).reshape(-1, 1)
        return inputs, labels


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--cm_threshold", type=float, default=0.5)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", type=str, default=str(Path.home() / "seizure-detector" / "results" / "confusion_matrices"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    meta = load_meta(data_dir)
    views = meta["views"]

    X, y = load_arrays(data_dir, views)
    N = int(y.shape[0])
    if N == 0:
        raise RuntimeError("Empty dataset.")

    _, va_idx = split_indices(N, args.val_split, args.seed)
    val_seq = MemmapSequence(X, y, views, va_idx, args.batch_size)

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    for inputs, labels in val_seq:
        probs = model.predict(inputs, verbose=0)
        preds = (probs >= args.cm_threshold).astype(int)
        y_true.extend(labels.reshape(-1).tolist())
        y_pred.extend(preds.reshape(-1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    exp_name = model_path.parent.name  # e.g. top_side
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"confusion_matrix_val_thr{args.cm_threshold:.2f}.csv"
    cm_df.to_csv(out_path, index=True)

    # also save metadata for reproducibility
    info = {
        "data_dir": str(data_dir),
        "model_path": str(model_path),
        "views": views,
        "N": N,
        "val_split": args.val_split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "cm_threshold": args.cm_threshold,
        "out_path": str(out_path),
        "confusion_matrix": cm,
    }
    with open(out_dir / f"confusion_matrix_val_thr{args.cm_threshold:.2f}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("[OK] Saved:")
    print(f"  - {out_path}")
    print(f"  - {out_dir / f'confusion_matrix_val_thr{args.cm_threshold:.2f}.json'}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
