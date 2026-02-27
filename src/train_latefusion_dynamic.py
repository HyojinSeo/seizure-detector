#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "processed_frames" / "sessions"


def list_tokens(frames_dir: Path) -> List[str]:
    tokens = set()
    for p in frames_dir.glob("y_FRAMES_*.npy"):
        m = re.match(r"y_FRAMES_(.+)\.npy$", p.name)
        if m:
            tokens.add(m.group(1))
    return sorted(tokens)


def load_frames_mmap(token: str, view: str) -> np.ndarray:
    p = FRAMES_DIR / f"X_FRAMES_{view}_{token}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing frames: {p}")
    return np.load(p, mmap_mode="r")  # uint8 (N,H,W,1)


def load_y(token: str) -> np.ndarray:
    p = FRAMES_DIR / f"y_FRAMES_{token}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing y: {p}")
    return np.load(p, mmap_mode="r").astype(np.uint8).reshape(-1)


def iter_windows_for_token(
    token: str,
    views: List[str],
    seq_len: int,
    stride: int,
    label_mode: str = "any",
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Yields:
      ({"TOP": (T,H,W,1) float32 0..1, ...}, y_seq (1,) int64)
    """
    Xv = {v: load_frames_mmap(token, v) for v in views}
    y = load_y(token)
    Ns = [int(Xv[v].shape[0]) for v in views] + [int(y.shape[0])]
    nmin = min(Ns)
    if len(set(Ns)) != 1:
        # trim mismatch
        for v in views:
            Xv[v] = Xv[v][:nmin]
        y = y[:nmin]

    N = nmin
    if N < seq_len:
        return

    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len

        # label
        y_win = y[start:end]
        if label_mode == "any":
            lab = int(np.any(y_win > 0))
        elif label_mode == "center":
            lab = int(y[start + seq_len // 2])
        elif label_mode == "majority":
            lab = int(np.sum(y_win > 0) > (seq_len / 2.0))
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        # inputs dict: float32 scaled 0..1
        x_out: Dict[str, np.ndarray] = {}
        for v in views:
            # uint8 -> float32 0..1
            x = np.array(Xv[v][start:end], dtype=np.float32) / 255.0
            x_out[v] = x

        yield x_out, np.array(lab, dtype=np.int64)


def make_dataset(
    tokens: List[str],
    views: List[str],
    seq_len: int,
    stride: int,
    label_mode: str,
    batch_size: int,
    shuffle_buffer: int = 2048,
) -> tf.data.Dataset:
    # output signature
    input_sig = {
        v: tf.TensorSpec(shape=(seq_len, 128, 128, 1), dtype=tf.float32) for v in views
    }
    out_sig = (input_sig, tf.TensorSpec(shape=(), dtype=tf.int64))

    def gen():
        for tok in tokens:
            yield from iter_windows_for_token(tok, views, seq_len, stride, label_mode=label_mode)

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(views: List[str], seq_len: int) -> tf.keras.Model:
    """
    예시용 간단 late-fusion (각 view를 작은 CNN으로 인코딩 후 concat)
    기존 all3 모델 아키텍처가 이미 있다면 그걸 그대로 가져오시는 걸 추천합니다.
    """
    inputs = {}
    feats = []

    for v in views:
        inp = tf.keras.Input(shape=(seq_len, 128, 128, 1), name=v)
        # TimeDistributed CNN
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"))(inp)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.GRU(64)(x)  # sequence encoder
        inputs[v] = inp
        feats.append(x)

    if len(feats) > 1:
        x = tf.keras.layers.Concatenate()(feats)
    else:
        x = feats[0]

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True, type=str)
    ap.add_argument("--views", nargs="+", default=["TOP", "SIDE", "SIDE2"], choices=["TOP", "SIDE", "SIDE2"])
    ap.add_argument("--seq_len", required=True, type=int)
    ap.add_argument("--stride", required=True, type=int)
    ap.add_argument("--label_mode", default="any", choices=["any", "center", "majority"])
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--include_tokens", nargs="*", default=None, help="Optional explicit token list (010626F1 etc.)")
    args = ap.parse_args()

    tokens = list_tokens(FRAMES_DIR)
    if args.include_tokens:
        allow = set(args.include_tokens)
        tokens = [t for t in tokens if t in allow]

    if not tokens:
        raise RuntimeError("No tokens found.")

    print(f"[INFO] tokens={len(tokens)}, views={args.views}, seq_len={args.seq_len}, stride={args.stride}")

    # NOTE: 여기서 train/val split은 간단히 90/10 예시
    n = len(tokens)
    n_train = max(1, int(round(n * 0.9)))
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:] if n_train < n else tokens[-1:]

    ds_train = make_dataset(train_tokens, args.views, args.seq_len, args.stride, args.label_mode, args.batch_size)
    ds_val = make_dataset(val_tokens, args.views, args.seq_len, args.stride, args.label_mode, args.batch_size)

    model = build_model(args.views, args.seq_len)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    out_dir = PROJECT_ROOT / "results" / "latefusion" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best_model.keras"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=[ckpt])

    # save final
    model.save(str(out_dir / "final_model.keras"))

    # save args/history
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {out_dir}")


if __name__ == "__main__":
    main()
