#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()

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
    n_steps = int(duration_sec * FPS_TARGET)
    for _ in tqdm(range(n_steps), desc=f"Extracting {video_path.name}"):
        t = _ / FPS_TARGET
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, RESIZE_SHAPE)
        frames.append(gray_resized)

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
    return np.stack(seqs, axis=0).astype(np.float32)  # (Nseq,T,H,W,1)


def find_view_files_by_session(raw_video_dir: Path, session: str) -> Dict[str, Path]:
    """
    session examples:
      "KA010626 M2"
      "KA010626 M2 B"

    Searches raw_video_dir for:
      POST KA010626 M2 ... -webcamup(.mp4)
      POST KA010626 M2 ... -webcamside1(.mp4)
      POST KA010626 M2 ... -webcamside2(.mp4)
    """
    s = session.strip().upper()

    # allow user to type with/without KA prefix
    if not s.startswith("KA"):
        s = "KA" + s

    # Split like: KA010626 M2 B -> date=010626, animal=M2, booster=True
    parts = s.split()
    if len(parts) < 2:
        raise ValueError('Session must look like "KA010626 M2" or "KA010626 M2 B"')

    date = parts[0].replace("KA", "")
    animal = parts[1]
    booster = (len(parts) >= 3 and parts[2] == "B")

    # Build a tolerant matcher (spaces/dashes don’t matter)
    def is_match(p: Path, view_sub: str) -> bool:
        name = p.name.lower()

        if not name.endswith(".mp4"):
            return False
        if "post ka" not in name:
            return False
        if date.lower() not in name:
            return False
        if animal.lower() not in name:
            return False
        if view_sub not in name:
            return False

        if booster:
            # require booster indicator near animal (e.g., "M2 B")
            if f"{animal.lower()} b" not in name:
                return False
        else:
            # avoid mixing booster files
            if f"{animal.lower()} b" in name:
                return False

        return True

    out: Dict[str, Path] = {}
    for v, sub in VIEW_SUBSTR.items():
        cand = [
            p for p in raw_video_dir.iterdir()
            if p.is_file() and is_match(p, sub)
        ]

        if len(cand) != 1:
            raise RuntimeError(
                f"Expected exactly 1 match for {v} in {raw_video_dir} "
                f"(session={session}). Found {len(cand)}: "
                f"{[c.name for c in cand[:5]]}"
            )

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
    #ap.add_argument("--input_dir", required=True, type=str, help="Folder containing TOP/SIDE/SIDE2 mp4 for one session")
    ap.add_argument("--session", required=True, type=str, help='Session like "KA010626 M2" or "KA010626 M2 B"')
    ap.add_argument("--raw_video_dir", type=str, default=str(RAW_VIDEO_DIR), help="Where to search videos (default: ~/gcs/inputs)")
    ap.add_argument("--model_path", required=True, type=str, help="Path to best_model.keras")
    ap.add_argument("--out_xlsx", required=True, type=str, help="Output Excel (.xlsx) path for seizure intervals")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    raw_video_dir = Path(args.raw_video_dir).expanduser()
    view_files = find_view_files_by_session(raw_video_dir, args.session)

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
    if len(df_intervals) == 0:
        print("[OK] No seizure intervals detected with the given threshold.")
    else:
        df_intervals["start_hms"] = pd.to_timedelta(df_intervals["start_sec"], unit="s")
        df_intervals["end_hms"] = pd.to_timedelta(df_intervals["end_sec"], unit="s")

    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_intervals.to_excel(out_path, index=False)
    print(f"[OK] Saved intervals Excel: {out_path}")
    print(df_intervals.head(20))


if __name__ == "__main__":
    main()
