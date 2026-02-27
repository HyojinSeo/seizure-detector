#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer seizure timeline intervals from raw multi-view videos (TOP/SIDE/SIDE2)
without preprocessing to .npy.

Key points:
- seq_len / stride are configurable via CLI
- if --seq_len is not provided, we infer it from the model input shape (T dimension)
- intervals are computed using the SAME stride/seq_len used for sequence creation

Example:
  python src/infer_timeline_latefusion.py \
    --session "KA010626 F1 B" \
    --model_path results/latefusion/all3/best_model.keras \
    --threshold 0.50 \
    --stride 4 \
    --out_xlsx results/infer/010626/all3/010626_F1_B_pred.xlsx
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()

FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)

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
    N = int(X_frames.shape[0])
    if N < seq_len:
        raise RuntimeError(f"Not enough frames: N={N} < seq_len={seq_len}")

    seqs = []
    for start in range(0, N - seq_len + 1, stride):
        seqs.append(X_frames[start : start + seq_len])

    if not seqs:
        raise RuntimeError(f"No sequences created. N={N}, seq_len={seq_len}, stride={stride}")

    return np.stack(seqs, axis=0).astype(np.float32)  # (Nseq,T,H,W,1)


def parse_session(session: str) -> Tuple[str, str, bool, str]:
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
        booster_token = parts[2]
        if booster_token == "B" and len(parts) >= 4 and re.match(r"^\d+$", parts[3]):
            booster_token = f"B-{parts[3]}"
        booster_token = booster_token.replace("_", "-")

    return date, animal, booster, booster_token


def find_view_files_by_session(raw_video_dir: Path, session: str) -> Dict[str, Path]:
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

        if booster:
            if f"{animal.lower()} b" not in name:
                return False
            if booster_token and booster_token != "B":
                bt = booster_token.lower()
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


def infer_seq_len_from_model(model: tf.keras.Model) -> Optional[int]:
    """
    Works when model input is like dict:
      TOP: (None, T, H, W, C)
      SIDE: (None, T, H, W, C) ...
    """
    try:
        if isinstance(model.input_shape, dict):
            first_key = list(model.input_shape.keys())[0]
            shp = model.input_shape[first_key]
        else:
            shp = model.input_shape
        # shp: (None, T, H, W, C)
        if shp and len(shp) >= 2 and shp[1] is not None:
            return int(shp[1])
    except Exception:
        pass
    return None


def probs_to_intervals(probs: np.ndarray, threshold: float, stride: int, seq_len: int) -> List[Tuple[float, float, float]]:
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
        start = i * stride
        end = (j - 1) * stride + seq_len
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
    ap.add_argument("--session", required=True, type=str)
    ap.add_argument("--raw_video_dir", type=str, default=str(RAW_VIDEO_DIR))
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--out_xlsx", required=True, type=str)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--views", nargs="+", default=["TOP", "SIDE", "SIDE2"], choices=["TOP", "SIDE", "SIDE2"])
    ap.add_argument("--stride", type=int, default=4, help="Stride in seconds (e.g., 4). Must match how you trained sequences.")
    ap.add_argument("--seq_len", type=int, default=None, help="Window length in seconds/frames (T). If omitted, inferred from model.")
    args = ap.parse_args()

    raw_video_dir = Path(args.raw_video_dir).expanduser()
    model_path = Path(args.model_path)

    if not raw_video_dir.exists():
        raise FileNotFoundError(f"raw_video_dir not found: {raw_video_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    view_files_all = find_view_files_by_session(raw_video_dir, args.session)
    views = [v for v in ["TOP", "SIDE", "SIDE2"] if v in args.views]
    view_files = {v: view_files_all[v] for v in views}

    print("Using files:")
    for v in views:
        print(f"  - {v}: {view_files[v].name}")

    model = tf.keras.models.load_model(str(model_path))

    seq_len = args.seq_len
    if seq_len is None:
        inferred = infer_seq_len_from_model(model)
        if inferred is None:
            raise RuntimeError("Could not infer seq_len from model. Please pass --seq_len explicitly.")
        seq_len = inferred

    stride = int(args.stride)

    print(f"[INFO] Using seq_len={seq_len}, stride={stride}, threshold={args.threshold}")

    X_frames = {v: extract_frames_1fps_gray(view_files[v]) for v in views}
    X_seq = {v: make_sequences(X_frames[v], seq_len=seq_len, stride=stride) for v in views}

    nseqs = {v: int(X_seq[v].shape[0]) for v in views}
    nmin = min(nseqs.values())
    if len(set(nseqs.values())) != 1:
        print(f"[WARN] Nseq mismatch across views {nseqs} -> trimming to {nmin}")
        for v in views:
            X_seq[v] = X_seq[v][:nmin]

    probs = model.predict(X_seq, batch_size=args.batch_size, verbose=1).reshape(-1)
    intervals = probs_to_intervals(probs, threshold=float(args.threshold), stride=stride, seq_len=seq_len)

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
