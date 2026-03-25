#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer seizure window-runs from raw multi-view videos (TOP/SIDE/SIDE2)
using a trained late-fusion model, without preprocessing to .npy.

Debug version:
- Prints detailed video metadata per view
- Prints extracted frame counts and approximate end times
- Prints sequence counts and approximate end times
- Prints whether cross-view trimming happened
- Prints final prediction coverage window range
- Optionally saves per-window probabilities as CSV
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


def sec_to_hhmmss(sec: float) -> str:
    sec_int = int(round(float(sec)))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_metadata(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0
    cap.release()

    return {
        "native_fps": native_fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def extract_frames_1fps_gray(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0

    if duration_sec <= 0:
        raise RuntimeError(
            f"Bad duration for video: {video_path} "
            f"(fps={native_fps}, frames={frame_count})"
        )

    print(
        f"[DEBUG] {video_path.name}: fps={native_fps:.6f}, "
        f"frame_count={int(frame_count)}, duration_sec={duration_sec:.2f} "
        f"({sec_to_hhmmss(duration_sec)})"
    )

    frames: List[np.ndarray] = []
    n_steps = int(duration_sec * FPS_TARGET)
    last_ok_t = None
    first_fail_t = None

    for i in tqdm(range(n_steps), desc=f"Extracting {video_path.name}"):
        t = i / FPS_TARGET
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            first_fail_t = t
            print(
                f"[WARN] read() failed for {video_path.name} at requested t={t:.2f}s "
                f"({sec_to_hhmmss(t)}). Stopping extraction."
            )
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, RESIZE_SHAPE)
        frames.append(gray_resized)
        last_ok_t = t

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    extracted_n = len(frames)
    approx_last_sec = extracted_n - 1
    print(
        f"[DEBUG] {video_path.name}: requested_steps={n_steps}, extracted_frames={extracted_n}, "
        f"approx_last_extracted_sec={approx_last_sec} ({sec_to_hhmmss(approx_last_sec)})"
    )
    if last_ok_t is not None:
        print(
            f"[DEBUG] {video_path.name}: last successful read at t={last_ok_t:.2f}s "
            f"({sec_to_hhmmss(last_ok_t)})"
        )
    if first_fail_t is not None:
        print(
            f"[DEBUG] {video_path.name}: first failed read at t={first_fail_t:.2f}s "
            f"({sec_to_hhmmss(first_fail_t)})"
        )

    x = np.stack(frames, axis=0).astype(np.float32)
    x = x / 255.0
    x = x[..., np.newaxis]
    return x


def make_sequences(x_frames: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    n = int(x_frames.shape[0])
    if n < seq_len:
        raise RuntimeError(f"Not enough frames: N={n} < seq_len={seq_len}")

    seqs = []
    for start in range(0, n - seq_len + 1, stride):
        seqs.append(x_frames[start : start + seq_len])

    if not seqs:
        raise RuntimeError(f"No sequences created. N={n}, seq_len={seq_len}, stride={stride}")

    return np.stack(seqs, axis=0).astype(np.float32)


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
    for view, sub in VIEW_SUBSTR.items():
        cand = [p for p in raw_video_dir.iterdir() if p.is_file() and is_match(p, sub)]
        if len(cand) != 1:
            preview = [c.name for c in cand[:10]]
            raise RuntimeError(
                f"Expected exactly 1 match for view={view} in {raw_video_dir} "
                f"(session={session}). Found {len(cand)} matches: {preview}"
            )
        out[view] = cand[0]

    return out


def infer_seq_len_from_model(model: tf.keras.Model) -> Optional[int]:
    try:
        if isinstance(model.input_shape, dict):
            first_key = list(model.input_shape.keys())[0]
            shp = model.input_shape[first_key]
        else:
            shp = model.input_shape

        if shp and len(shp) >= 2 and shp[1] is not None:
            return int(shp[1])
    except Exception:
        pass
    return None


def probs_to_window_runs(
    probs: np.ndarray,
    threshold: float,
    stride: int,
    seq_len: int,
) -> List[Dict[str, float]]:
    pred = (probs >= threshold).astype(np.int32)
    rows: List[Dict[str, float]] = []

    i = 0
    while i < len(pred):
        if pred[i] == 0:
            i += 1
            continue

        j = i
        while j < len(pred) and pred[j] == 1:
            j += 1

        start_window = int(i)
        end_window = int(j - 1)
        run_probs = probs[i:j]

        mean_prob = float(np.mean(run_probs))
        max_prob = float(np.max(run_probs))
        num_windows = int(j - i)
        start_sec_approx = float(start_window * stride)
        end_sec_approx = float(end_window * stride + seq_len)

        rows.append(
            {
                "start_window": start_window,
                "end_window": end_window,
                "num_windows": num_windows,
                "mean_prob": mean_prob,
                "max_prob": max_prob,
                "start_sec_approx": start_sec_approx,
                "end_sec_approx": end_sec_approx,
                "start_hms_approx": sec_to_hhmmss(start_sec_approx),
                "end_hms_approx": sec_to_hhmmss(end_sec_approx),
            }
        )
        i = j

    return rows


def save_window_probs(
    probs: np.ndarray,
    threshold: float,
    stride: int,
    seq_len: int,
    out_csv: Path,
) -> None:
    rows = []
    for i, p in enumerate(probs.tolist()):
        rows.append(
            {
                "window_index": int(i),
                "prob": float(p),
                "pred_label": int(p >= threshold),
                "start_sec_approx": float(i * stride),
                "end_sec_approx": float(i * stride + seq_len),
                "start_hms_approx": sec_to_hhmmss(i * stride),
                "end_hms_approx": sec_to_hhmmss(i * stride + seq_len),
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def print_view_debug_summary(x_frames: Dict[str, np.ndarray], x_seq: Dict[str, np.ndarray], seq_len: int, stride: int) -> None:
    print("\n[DEBUG] Per-view extracted frame summary")
    for v in x_frames:
        nframes = int(x_frames[v].shape[0])
        approx_last_frame_sec = nframes - 1
        print(
            f"  {v}: frames={nframes}, approx_last_frame_sec={approx_last_frame_sec} "
            f"({sec_to_hhmmss(approx_last_frame_sec)})"
        )

    print("\n[DEBUG] Per-view sequence summary")
    for v in x_seq:
        nseq = int(x_seq[v].shape[0])
        approx_last_window_start = (nseq - 1) * stride
        approx_last_window_end = approx_last_window_start + seq_len
        print(
            f"  {v}: nseq={nseq}, last_window_start≈{approx_last_window_start}s "
            f"({sec_to_hhmmss(approx_last_window_start)}), "
            f"last_window_end≈{approx_last_window_end}s "
            f"({sec_to_hhmmss(approx_last_window_end)})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=str)
    ap.add_argument("--raw_video_dir", type=str, default=str(RAW_VIDEO_DIR))
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--out_xlsx", required=True, type=str)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--views", nargs="+", default=["TOP", "SIDE", "SIDE2"], choices=["TOP", "SIDE", "SIDE2"])
    ap.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride in frames/seconds at 1 fps. Must match training setup.",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Window length T. If omitted, inferred from model input shape.",
    )
    ap.add_argument(
        "--save_window_probs_csv",
        action="store_true",
        help="Also save per-window probabilities as a CSV next to out_xlsx.",
    )
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

    print("\n[DEBUG] Raw video metadata")
    for v in views:
        meta = get_video_metadata(view_files[v])
        print(
            f"  {v}: size={meta['width']}x{meta['height']}, fps={meta['native_fps']:.6f}, "
            f"frames={int(meta['frame_count'])}, duration={meta['duration_sec']:.2f}s "
            f"({sec_to_hhmmss(meta['duration_sec'])})"
        )

    model = tf.keras.models.load_model(str(model_path), compile=False)

    seq_len = args.seq_len
    if seq_len is None:
        inferred = infer_seq_len_from_model(model)
        if inferred is None:
            raise RuntimeError("Could not infer seq_len from model. Please pass --seq_len explicitly.")
        seq_len = inferred

    stride = int(args.stride)

    print(f"\n[INFO] Using seq_len={seq_len}, stride={stride}, threshold={args.threshold}")

    x_frames = {v: extract_frames_1fps_gray(view_files[v]) for v in views}
    x_seq = {v: make_sequences(x_frames[v], seq_len=seq_len, stride=stride) for v in views}

    print_view_debug_summary(x_frames, x_seq, seq_len=seq_len, stride=stride)

    nseqs = {v: int(x_seq[v].shape[0]) for v in views}
    nmin = min(nseqs.values())
    if len(set(nseqs.values())) != 1:
        print(f"\n[WARN] Nseq mismatch across views {nseqs} -> trimming to {nmin}")
        print("[WARN] This means the shortest view determines how far inference can go.")
        for v in views:
            before = int(x_seq[v].shape[0])
            x_seq[v] = x_seq[v][:nmin]
            after = int(x_seq[v].shape[0])
            last_window_end = (after - 1) * stride + seq_len
            print(
                f"  [TRIM] {v}: {before} -> {after} sequences, "
                f"trimmed_last_window_end≈{last_window_end}s "
                f"({sec_to_hhmmss(last_window_end)})"
            )
    else:
        last_window_end = (nmin - 1) * stride + seq_len
        print(
            f"\n[DEBUG] No cross-view trimming needed. Common last_window_end≈{last_window_end}s "
            f"({sec_to_hhmmss(last_window_end)})"
        )

    probs = model.predict(x_seq, batch_size=args.batch_size, verbose=1).reshape(-1)

    if len(probs) > 0:
        final_window_index = len(probs) - 1
        final_start_sec = final_window_index * stride
        final_end_sec = final_start_sec + seq_len
        print(
            f"\n[DEBUG] Prediction coverage: total_windows={len(probs)}, "
            f"final_window_index={final_window_index}, "
            f"final_window_start≈{final_start_sec}s ({sec_to_hhmmss(final_start_sec)}), "
            f"final_window_end≈{final_end_sec}s ({sec_to_hhmmss(final_end_sec)})"
        )
        print(
            f"[DEBUG] Probability stats: min={float(np.min(probs)):.6f}, "
            f"max={float(np.max(probs)):.6f}, mean={float(np.mean(probs)):.6f}"
        )

    run_rows = probs_to_window_runs(
        probs=probs,
        threshold=float(args.threshold),
        stride=stride,
        seq_len=seq_len,
    )

    df_runs = pd.DataFrame(
        run_rows,
        columns=[
            "start_window",
            "end_window",
            "num_windows",
            "mean_prob",
            "max_prob",
            "start_sec_approx",
            "end_sec_approx",
            "start_hms_approx",
            "end_hms_approx",
        ],
    )

    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_runs.to_excel(out_path, index=False)

    print(f"\n[OK] Saved window-run Excel: {out_path}")
    print(f"[INFO] Total windows: {len(probs)}")
    print(f"[INFO] Positive windows >= {args.threshold}: {int(np.sum(probs >= args.threshold))}")
    print(f"[INFO] Number of positive runs: {len(df_runs)}")

    if df_runs.empty:
        print("[OK] No seizure window-runs detected with the given threshold.")
    else:
        print(df_runs.head(20).to_string(index=False))
        print("\n[DEBUG] Last 20 positive runs:")
        print(df_runs.tail(20).to_string(index=False))

    if args.save_window_probs_csv:
        csv_path = out_path.with_name(out_path.stem + "_window_probs.csv")
        save_window_probs(
            probs=probs,
            threshold=float(args.threshold),
            stride=stride,
            seq_len=seq_len,
            out_csv=csv_path,
        )
        print(f"[OK] Saved per-window probability CSV: {csv_path}")


if __name__ == "__main__":
    main()
