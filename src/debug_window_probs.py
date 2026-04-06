#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug window probabilities for a specific time range.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from infer_timeline_latefusion import (
    find_view_files_by_session,
    extract_frames_1fps_gray,
    make_sequences,
)

RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()

SESSION = "010626 F1"
MODEL_PATH = "results/latefusion/all3_16_4/best_model.keras"

STRIDE = 4
SEQ_LEN = 16

START_SEC = 2890
END_SEC = 2940


def main():

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Finding video files...")
    view_files = find_view_files_by_session(RAW_VIDEO_DIR, SESSION)

    print("Extracting frames...")
    X_frames = {v: extract_frames_1fps_gray(view_files[v]) for v in view_files}

    print("Making sequences...")
    X_seq = {v: make_sequences(X_frames[v], seq_len=SEQ_LEN, stride=STRIDE) for v in view_files}

    # trim to equal length
    nmin = min(X_seq[v].shape[0] for v in X_seq)
    for v in X_seq:
        X_seq[v] = X_seq[v][:nmin]

    print("Running prediction...")
    probs = model.predict(X_seq, batch_size=16, verbose=1).reshape(-1)

    start_w = int(START_SEC / STRIDE)
    end_w = int(END_SEC / STRIDE)

    print("\nWindow index range:", start_w, "to", end_w)
    print("Window probabilities:\n")

    for i in range(start_w, min(end_w + 1, len(probs))):
        start_time = i * STRIDE
        print(
            f"window {i:4d} | start_sec={start_time:5.0f} | prob={probs[i]:.4f}"
        )


if __name__ == "__main__":
    main()
