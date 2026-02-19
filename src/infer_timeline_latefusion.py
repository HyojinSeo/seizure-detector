#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer seizure timeline intervals from raw multi-view videos (TOP/SIDE/SIDE2)
without preprocessing to .npy.

It:
  1) finds mp4 files for a given session in raw_video_dir
  2) extracts 1 fps grayscale frames (128x128)
  3) builds sequences (SEQ_LEN=16, STRIDE=4)
  4) runs a trained late-fusion model (expects dict inputs: TOP/SIDE/SIDE2)
  5) converts probabilities to merged positive intervals
  6) saves intervals to Excel

Example:
  python src/infer_timeline_latefusion.py \
    --session "KA010626 F1 B" \
    --model_path results/latefusion/all3/best_model.keras \
    --threshold 0.50 \
    --out_xlsx results/infer/010626/all3/010626_F1_B_pred.xlsx
"""

import argparse
import re
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

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0

    if duration_sec <= 0:
        raise RuntimeError(f"Bad duration for video: {video_path} (fps={native_fps}, frames={frame_count})")

    frames: List[np.ndarray] = []
    n_steps = int(duration_sec * FPS_TARGET)

    for i in tqdm(range(n_steps), desc=f"Extracting {video_path.name}"):
        t = i / FPS_TARGET
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
    X = X / 255.0
    X = X[..., np.newaxis]  # (N,H,W,1)
    return X


def make_sequences(X_frames: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    # X_frames: (N,H,W,1)
    N = int(X_frames.shape[0])
    if N < seq_len:
        raise RuntimeError(f"Not enough frames: N={N} < seq_len={seq_len}")

    seqs = []
    for start in range(0, N - seq_len + 1, stride):
        seqs.append(X_frames[start : start + seq_len])  # (T,H,W,1)

    if not seqs:
        raise RuntimeError(f"No sequences created. N={N}, seq_len={seq_len}, stride={stride}")

    return np.stack(seqs, axis=0).astype(np.float32)  # (Nseq,T,H,W,1)


def parse_session(session: str) -> Tuple[str, str, bool, str]:
    """
    Returns:
      date (e.g., "010626"),
      animal (e.g., "F1"),
      booster (True/False),
      booster_token (normalized booster string or "")
        - ""          : non-booster
        - "B"         : booster without number
        - "B-1"/"B-2" : booster with number
    Accepts:
      "KA010626 F1"
      "KA010626 F1 B"
      "KA010626 F1 B-1"
      "010626 F1" (KA optional)
    """
    s = session.strip().upper()
    if not s.startswith("KA"):
        s = "KA" + s

    parts = s.split()
    if len(parts) < 2:
        raise ValueError('Session must look like "KA010626 F1" or "KA010626 F1 B"')

    date = parts[0].replace("KA", "")
    animal = parts[1]

    booster = False
    booster_token = ""
    if len(parts) >= 3 and parts[2].startswith("B"):
        booster = True
        booster_token = parts[2]  # "B" or "B-1" etc.

        # normalize a few patterns like "B_1" or "B 1"
        if booster_token == "B" and len(parts) >= 4 and re.match(r"^\d+$", parts[3]):
            booster_token = f"B-{parts[3]}"
        booster_token = booster_token.replace("_", "-")

    return date, animal, booster, booster_token


def find_view_files_by_session(raw_video_dir: Path, session: str) -> Dict[str, Path]:
    """
    Find exactly one mp4 for each view (TOP/SIDE/SIDE2) in raw_video_dir.
    Matching is tolerant to extra words, as long as it includes:
      - "post ka"
      - date string (e.g., 010626)
      - animal string (e.g., f1)
      - view substring (webcamup/webcamside1/webcamside2)
      - booster inclusion/exclusion rule
    """
    date, animal, booster, booster_token = parse_session(session)

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

        # Booster filtering:
        # - if booster: must contain "<animal> b"
        # - if non-booster: must NOT contain "<animal> b"
        if booster:
            if f"{animal.lower()} b" not in name:
                return False
            # If user specified B-1 / B-2, require that token
            if booster_token and booster_token != "B":
                # accept "b-1" or "b 1" variations loosely
                bt = booster_token.lower()  # e.g., "b-1"
                # allow either "b-1" or "b 1" to match
                num = bt.split("-", 1)[1] if "-" in bt else ""
                ok = (bt in name) or (num and f"b {num}" in name)
                if not ok:
                    return False
        else:
            if f"{animal.lower()} b" in name:
                return False

        return True

    out: Dict[str, Path] = {}
    for v, sub in VIEW_SUBSTR.items():
        cand = [p for p in raw_video_dir.iterdir() if p.is_file() and is_match(p, sub)]
        if len(cand) != 1:
            preview = [c.name for c in cand[:10]]
            raise RuntimeError(
                f"Expected exactly 1 match for view={v} in {raw_video_dir} (session={session}). "
                f"Found {len(cand)} matches: {preview}"
            )
        out[v] = cand[0]

    return out


def probs_to_intervals(probs: np.ndarray, threshold: float) -> List[Tuple[float, float, float]]:
    """
    Return merged intervals: (start_sec, end_sec, mean_prob) for consecutive positives.
    Each sequence i corresponds to [i*STRIDE, i*STRIDE + SEQ_LEN).
    """
    pred = (probs >= threshold).astype(np.int32)
    intervals: List[Tuple[float, float, float]] = []

    i = 0
    while i < len(pred):
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < len(pred) and pred[j] == 1:
            j += 1
        start = i * STRIDE
        end = (j - 1) * STRIDE + SEQ_LEN
        mean_p = float(np.mean(probs[i:j]))
        intervals.append((float(start), float(end), mean_p))
        i = j

    return intervals


def sec_to_hhmmss(sec: float) -> str:
    sec_int = int(round(float(sec)))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=str, help='e.g., "KA010626 F1", "KA010626 F1 B", "KA010626 F1 B-1"')
    ap.add_argument("--raw_video_dir", type=str, default=str(RAW_VIDEO_DIR), help="Default: ~/gcs/inputs")
    ap.add_argument("--model_path", required=True, type=str, help="Path to best_model.keras")
    ap.add_argument("--out_xlsx", required=True, type=str, help="Output Excel (.xlsx) path for seizure intervals")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--views", nargs="+", default=["TOP", "SIDE", "SIDE2"], choices=["TOP", "SIDE", "SIDE2"],
                    help="Which views to use (default: TOP SIDE SIDE2). Must match the trained model inputs.")
    args = ap.parse_args()

    raw_video_dir = Path(args.raw_video_dir).expanduser()
    model_path = Path(args.model_path)

    if not raw_video_dir.exists():
        raise FileNotFoundError(f"raw_video_dir not found: {raw_video_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    # Find files
    view_files_all = find_view_files_by_session(raw_video_dir, args.session)

    # Filter to requested views in a stable order
    views = [v for v in ["TOP", "SIDE", "SIDE2"] if v in args.views]
    view_files = {v: view_files_all[v] for v in views}

    print("Using files:")
    for v in views:
        print(f"  - {v}: {view_files[v].name}")

    # Extract frames
    X_frames = {v: extract_frames_1fps_gray(view_files[v]) for v in views}

    # Make sequences
    X_seq = {v: make_sequences(X_frames[v], SEQ_LEN, STRIDE) for v in views}

    # Trim to min Nseq across views
    nseqs = {v: int(X_seq[v].shape[0]) for v in views}
    nmin = min(nseqs.values())
    if len(set(nseqs.values())) != 1:
        print(f"[WARN] Nseq mismatch across views {nseqs} -> trimming to {nmin}")
        for v in views:
            X_seq[v] = X_seq[v][:nmin]

    # Predict
    model = tf.keras.models.load_model(str(model_path))
    probs = model.predict(X_seq, batch_size=args.batch_size, verbose=1).reshape(-1)

    # Convert to intervals
    intervals = probs_to_intervals(probs, threshold=float(args.threshold))
    df_intervals = pd.DataFrame(intervals, columns=["start_sec", "end_sec", "mean_prob"])

    if df_intervals.empty:
        print("[OK] No seizure intervals detected with the given threshold.")
    else:
        df_intervals["start_hms"] = df_intervals["start_sec"].apply(sec_to_hhmmss)
        df_intervals["end_hms"] = df_intervals["end_sec"].apply(sec_to_hhmmss)

    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_intervals.to_excel(out_path, index=False)

    print(f"[OK] Saved intervals Excel: {out_path}")
    print(df_intervals.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
