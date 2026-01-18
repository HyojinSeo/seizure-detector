#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)
SEQ_LEN = 16
STRIDE = 4


VIEW_SUBSTR = {
    "TOP": "webcamup",
    "SIDE": "webcamside1",
    "SIDE2": "webcamside2",
}


def extract_frames_1fps_gray(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0

    frames: List[np.ndarray] = []
    t = 0.0
    while t < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, RESIZE_SHAPE)
        frames.append(gray_resized)
        t += 1.0 / FPS_TARGET

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    X = np.stack(frames, axis=0).astype(np.float32)  # (N,H,W)
    # scale 0..255 -> 0..1
    X = X / 255.0
    # add channel
    X = X[..., np.newaxis]  # (N,H,W,1)
    return X


def make_sequences(X_frames: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    # X_frames: (N,H,W,1)
    N = X_frames.shape[0]
    seqs = []
    for start in range(0, N - seq_len + 1, stride):
        seqs.append(X_frames[start:start+seq_len])  # (T,H,W,1)
    if not seqs:
        raise RuntimeError(f"No sequences created. N={N}, seq_len={seq_len}")
    return np.stack(seqs, axis=0)  # (Nseq,T,H,W,1)


def find_view_files(input_dir: Path) -> Dict[str, Path]:
    files = list(input_dir.iterdir())
    out = {}
    for v, sub in VIEW_SUBSTR.items():
        cand = [p for p in files if p.is_file() and p.suffix.lower() == ".mp4" and sub in p.name.lower()]
        if len(cand) != 1:
            raise RuntimeError(f"Expected exactly 1 file for {v} containing '{sub}'. Found {len(cand)}.")
        out[v] = cand[0]
    return out


def probs_to_intervals(probs: np.ndarray, threshold: float) -> List[Tuple[float, float, float]]:
    """
    Return merged intervals: (start_sec, end_sec, mean_prob) for consecutive positives.
    Each sequence i corresponds to [i*STRIDE, i*STRIDE + SEQ_LEN).
    """
    pred = (probs >= threshold).astype(int)
    intervals = []
    i = 0
    while i < len(pred):
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < len(pred) and pred[j] == 1:
            j += 1
        # sequences i..j-1 are positive
        start = i * STRIDE
        end = (j - 1) * STRIDE + SEQ_LEN
        mean_p = float(np.mean(probs[i:j]))
        intervals.append((float(start), float(end), mean_p))
        i = j
    return intervals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing TOP/SIDE/SIDE2 mp4 for one session")
    ap.add_argument("--model_path", required=True, type=str, help="Path to best_model.keras")
    ap.add_argument("--out_csv", required=True, type=str, help="Output CSV for seizure intervals")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    view_files = find_view_files(input_dir)

    print("Using files:")
    for v, p in view_files.items():
        print(f"  - {v}: {p.name}")

    # Extract frames per view
    X_frames = {v: extract_frames_1fps_gray(p) for v, p in view_files.items()}

    # Make sequences per view
    X_seq = {v: make_sequences(X_frames[v], SEQ_LEN, STRIDE) for v in X_frames.keys()}

    # Trim to min Nseq across views
    nseqs = [X_seq[v].shape[0] for v in X_seq.keys()]
    nmin = min(nseqs)
    if len(set(nseqs)) != 1:
        print(f"[WARN] Nseq mismatch across views {dict(zip(X_seq.keys(), nseqs))} -> trimming to {nmin}")
        for v in X_seq.keys():
            X_seq[v] = X_seq[v][:nmin]

    # Predict
    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(X_seq, batch_size=args.batch_size, verbose=1).reshape(-1)  # (Nseq,)

    # Convert to intervals
    intervals = probs_to_intervals(probs, threshold=args.threshold)

    df_intervals = pd.DataFrame(intervals, columns=["start_sec", "end_sec", "mean_prob"])
    df_intervals["start_hms"] = pd.to_timedelta(df_intervals["start_sec"], unit="s")
    df_intervals["end_hms"] = pd.to_timedelta(df_intervals["end_sec"], unit="s")

    df_intervals.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved intervals CSV: {args.out_csv}")
    print(df_intervals.head(20))


if __name__ == "__main__":
    main()
