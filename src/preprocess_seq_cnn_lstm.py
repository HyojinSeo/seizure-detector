#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence preprocessing script for seizure-detector project.

End-to-end pipeline for the CNN + LSTM/GRU model:

- Reads multi-view POST KA videos from ~/gcs/inputs by default
- Matches sessions to Excel sheets (seizure_stage.xlsx)
- Extracts frames at 1 fps, grayscale, resized to 128x128
- Labels each frame as seizure/non-seizure from Excel time intervals
- Builds sliding-window sequences (T frames) per session and per view
- Saves per-session npy files for each view (X_SEQ_*, y_SEQ_*)

Supports booster sessions:
- _B, _B-1, _B-2 (from Excel sheets)
- "B", "B-1", "B-2" (from video filenames)

Filtering:
- Use env var SESSIONS="121125 F1_B,121325 M3_B-1" to run a subset.

Optional:
- SKIP_EXISTING=1 to skip session+view if output npy files already exist.

Examples:
  python src/preprocess_seq_cnn_lstm.py
  python src/preprocess_seq_cnn_lstm.py --seq_len 32 --stride 8
  python src/preprocess_seq_cnn_lstm.py --seq_len 32 --stride 8 --output_dir data/processed_seq/sessions_32_8
  python src/preprocess_seq_cnn_lstm.py --seq_len 32 --stride 8 --dtype uint8 --output_dir data/processed_seq/sessions_train_32_8_uint8
  python src/preprocess_seq_cnn_lstm.py --seq_len 32 --stride 8 --dtype float32 --output_dir data/processed_seq/sessions_train_32_8_float32
"""

import argparse
import os
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ======================
# Default configuration
# ======================

FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)

# Original setting for float pipeline:
# DEFAULT_DTYPE = np.float32
DEFAULT_DTYPE_STR = "uint8"

DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE = 4
DEFAULT_LABEL_MODE = "any"  # "any", "max", "center", "majority"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "data" / "seizure_stage.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_seq" / "sessions_train_32_8_uint8"

VIEWS = ["TOP", "SIDE", "SIDE2"]
VIEW_CONFIG = {
    "TOP": "webcamup",
    "SIDE": "webcamside1",
    "SIDE2": "webcamside2",
}

VALID_EXTS = (".mp4",)

SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "0").strip() in {"1", "true", "True", "YES", "yes"}


# ======================
# Helper functions: sheet/session naming
# ======================

def get_session_id_from_sheet(sheet_name: str) -> Optional[str]:
    """
    Convert sheet name into a session id string.

    Supports:
      - '112625F1'       -> '112625 F1'
      - '112625F1_B'     -> '112625 F1_B'
      - '121325M3_B-1'   -> '121325 M3_B-1'
      - '121325M3_B_2'   -> '121325 M3_B-2'
    """
    if not isinstance(sheet_name, str) or not sheet_name.strip():
        return None

    clean = sheet_name.replace(" ", "")
    m = re.match(
        r"^(\d{6})([MF]\d)(?:_B(?:[-_]?(\d+))?)?$",
        clean,
        re.IGNORECASE,
    )
    if not m:
        return None

    date = m.group(1)
    rat = m.group(2).upper()
    b_num = m.group(3)

    if b_num:
        return f"{date} {rat}_B-{b_num}"
    elif "_B" in clean.upper():
        return f"{date} {rat}_B"
    else:
        return f"{date} {rat}"


def normalize_session_for_filename(session_id: str) -> str:
    """
    Normalize session id like '112625 F1_B' to '112625F1_B' for filenames.
    Keeps hyphens for B-1 (e.g., '121325 M3_B-1' -> '121325M3_B-1').
    """
    return session_id.replace(" ", "")


# ======================
# Helper functions: time parsing and intervals
# ======================

def parse_hms_to_seconds(s: str) -> Optional[float]:
    """
    Parse a time string like "MM:SS" or "HH:MM:SS" into seconds.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            return m * 60 + sec
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return h * 3600 + m * 60 + sec
        return None
    except ValueError:
        return None


def parse_time_interval(interval_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a time interval string like "20:28 - 20:57" into (start_sec, end_sec).
    """
    if not isinstance(interval_str, str):
        return None, None
    text = interval_str.strip()
    if not text or "-" not in text:
        return None, None
    left, right = text.split("-", 1)
    start_sec = parse_hms_to_seconds(left.strip())
    end_sec = parse_hms_to_seconds(right.strip())
    return start_sec, end_sec


def load_seizure_intervals(excel_path: Path, sheet_name: str) -> List[Tuple[float, float]]:
    """
    Load seizure intervals from a specific sheet in Excel.

    Assumes a 'Time' column with entries like "20:28 - 20:57".
    Any valid interval row is treated as a seizure interval, regardless of Stage.
    """
    xls = pd.ExcelFile(excel_path)
    df = pd.read_excel(xls, sheet_name=sheet_name)

    if "Time" not in df.columns:
        print(f"[WARN] Sheet {sheet_name} has no 'Time' column.")
        return []

    intervals: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        time_cell = row["Time"]
        start_sec, end_sec = parse_time_interval(time_cell)
        if start_sec is None or end_sec is None:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        intervals.append((start_sec, end_sec))

    print(f"Loaded {len(intervals)} seizure intervals from sheet '{sheet_name}'.")
    return intervals


def label_time_by_intervals(t_sec: float, intervals: List[Tuple[float, float]]) -> int:
    """
    Return 1 if t_sec falls inside any seizure interval, else 0.
    """
    for start, end in intervals:
        if start <= t_sec <= end:
            return 1
    return 0


# ======================
# Helper functions: video and Excel scanning
# ======================

def scan_videos(raw_video_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan raw_video_dir for POST KA .mp4 videos and return a mapping:

        sessions[session_id][view] = video_path

    session_id example:
      - "112625 F1"
      - "112625 F1_B"
      - "121325 M3_B-1"
    """
    sessions: Dict[str, Dict[str, Path]] = {}

    pattern = re.compile(
        r"POST\s*KA(\d{6})\s*([MF]\d)",
        re.IGNORECASE,
    )

    print(f"\nScanning videos in: {raw_video_dir}")
    for path in raw_video_dir.iterdir():
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not lower.endswith(VALID_EXTS):
            continue
        if "post ka" not in lower:
            continue

        match = pattern.search(path.name)
        if not match:
            print(f"  [WARN] No POST KA session match in filename: {path.name}")
            continue

        date = match.group(1)
        rat = match.group(2).upper()

        booster_match = re.search(r"\bB(?:[-_]?(\d+))?\b", path.name, re.IGNORECASE)
        if booster_match:
            b_num = booster_match.group(1)
            if b_num:
                session_id = f"{date} {rat}_B-{b_num}"
            else:
                session_id = f"{date} {rat}_B"
        else:
            session_id = f"{date} {rat}"

        view = None
        for v_name, substr in VIEW_CONFIG.items():
            if substr in lower:
                view = v_name
                break
        if view is None:
            print(f"  [WARN] Could not determine view for file: {path.name}")
            continue

        sessions.setdefault(session_id, {})[view] = path

    if not sessions:
        print("  (no POST KA videos found)")

    print("\nDetected sessions from videos (session_id -> views):")
    for sid in sorted(sessions.keys()):
        views_str = ", ".join(sorted(sessions[sid].keys()))
        print(f"  - {sid}: {views_str}")

    return sessions


def scan_excel_sessions(excel_path: Path) -> Dict[str, str]:
    """
    Scan Excel file and map session_id -> sheet_name.
    """
    session_to_sheet: Dict[str, str] = {}

    if not excel_path.exists():
        print(f"[WARN] Excel file not found: {excel_path}")
        return session_to_sheet

    try:
        xls = pd.ExcelFile(excel_path)
        sheet_list = xls.sheet_names
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {e}")
        return session_to_sheet

    print("\nExcel sheets found:")
    for name in sheet_list:
        print(f"  - {name}")

    for sheet in sheet_list:
        session_id = get_session_id_from_sheet(sheet)
        if session_id is None:
            continue
        session_to_sheet[session_id] = sheet

    print("\nSession IDs inferred from Excel sheets:")
    for sid in sorted(session_to_sheet.keys()):
        print(f"  - {sid} (sheet: {session_to_sheet[sid]})")

    return session_to_sheet


def print_session_match_summary(
    video_sessions: Dict[str, Dict[str, Path]],
    excel_sessions: Dict[str, str],
) -> List[str]:
    """
    Print a summary showing which sessions have both videos and Excel sheets.
    """
    video_keys = set(video_sessions.keys())
    excel_keys = set(excel_sessions.keys())
    all_sessions = sorted(video_keys.union(excel_keys))

    print("\nVideo / Excel session match summary:")
    if not all_sessions:
        print("  No sessions found in either videos or Excel.")
        return []

    common: List[str] = []
    for sid in all_sessions:
        has_video = sid in video_keys
        has_sheet = sid in excel_keys
        sheet_name = excel_sessions.get(sid, None)

        if has_video and has_sheet:
            print(f"[OK]     {sid}  (sheet: {sheet_name})")
            common.append(sid)
        elif has_video and not has_sheet:
            print(f"[NO XLS] {sid}  (no matching sheet)")
        elif not has_video and has_sheet:
            print(f"[NO VID] {sid}  (sheet: {sheet_name}, no matching POST video)")

    print("")
    return common


# ======================
# Sequence building
# ======================

def make_sequences_from_frames(
    x_frames: np.ndarray,
    y_frames: np.ndarray,
    seq_len: int,
    stride: int,
    label_mode: str = "any",
    dtype_str: str = "uint8",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from frame-level arrays.

    x_frames : (N_frames, H, W) or (N_frames, H, W, C)
    y_frames : (N_frames,) or (N_frames, 1) with 0/1 labels.
    """
    X = x_frames
    y = y_frames

    if y.ndim > 1:
        y = y.reshape(-1)

    n_frames = X.shape[0]
    if n_frames != y.shape[0]:
        raise ValueError(f"X and y have different number of frames: {n_frames} vs {y.shape[0]}")

    if X.ndim == 3:
        X = X[..., np.newaxis]  # (N, H, W, 1)

    sequences: List[np.ndarray] = []
    labels: List[int] = []

    for start in range(0, n_frames - seq_len + 1, stride):
        end = start + seq_len
        x_seq = X[start:end]
        y_seq_frames = y[start:end]

        if label_mode == "any":
            label = int(np.any(y_seq_frames > 0))
        elif label_mode == "max":
            label = int(np.max(y_seq_frames))
        elif label_mode == "center":
            center_idx = start + seq_len // 2
            label = int(y[center_idx])
        elif label_mode == "majority":
            label = int(np.sum(y_seq_frames > 0) > (seq_len / 2.0))
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        sequences.append(x_seq)
        labels.append(label)

    if not sequences:
        raise ValueError(
            f"No sequences generated. Check seq_len={seq_len} and stride={stride} "
            f"for n_frames={n_frames}."
        )

    x_seq = np.stack(sequences, axis=0)

    if dtype_str == "float32":
        x_seq = x_seq.astype(np.float32) / 255.0
    elif dtype_str == "uint8":
        x_seq = x_seq.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype_str: {dtype_str}")

    y_seq = np.array(labels, dtype=np.int64)

    print(
        f"Built {x_seq.shape[0]} sequences of length {seq_len} "
        f"from {n_frames} frames (stride={stride}). "
        f"Positive sequences: {np.sum(y_seq == 1)} / {y_seq.shape[0]}"
    )

    return x_seq, y_seq


def process_session_view(
    session_id: str,
    view: str,
    video_path: Path,
    intervals: List[Tuple[float, float]],
    seq_len: int,
    stride: int,
    label_mode: str,
    output_dir: Path,
    dtype_str: str,
) -> Dict[str, int]:
    """
    For a given session and view:
    - Sample frames at 1 fps
    - Grayscale + resize
    - Label frames by intervals
    - Build sequences
    - Save npy files
    """
    norm_sess = normalize_session_for_filename(session_id)
    x_out = output_dir / f"X_SEQ_{view}_{norm_sess}.npy"
    y_out = output_dir / f"y_SEQ_{view}_{norm_sess}.npy"

    if SKIP_EXISTING and x_out.exists() and y_out.exists():
        print(f"\n[SKIP] Outputs exist for {session_id} {view}: {x_out.name}, {y_out.name}")
        return {
            "session_id": session_id,
            "view": view,
            "frames": 0,
            "seq_samples": 0,
            "pos_frames": 0,
            "pos_sequences": 0,
            "skipped": 1,
        }

    print(f"\nProcessing session {session_id}, view {view}")
    print(f"  Video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0

    print(
        f"  Native FPS: {native_fps:.3f}, "
        f"frame_count: {frame_count}, "
        f"duration_sec: {duration_sec:.2f}"
    )

    frames: List[np.ndarray] = []
    labels: List[int] = []

    t = 0.0
    while t < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, RESIZE_SHAPE)
        label = label_time_by_intervals(t, intervals)

        frames.append(gray_resized)
        labels.append(label)

        t += 1.0 / FPS_TARGET

    cap.release()

    if not frames:
        print(f"[WARN] No frames extracted for {session_id} {view}")
        return {
            "session_id": session_id,
            "view": view,
            "frames": 0,
            "seq_samples": 0,
            "pos_frames": 0,
            "pos_sequences": 0,
            "skipped": 0,
        }

    frames_arr = np.stack(frames, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    print(
        f"  Collected {frames_arr.shape[0]} frames. "
        f"Positive frames: {np.sum(labels_arr == 1)}"
    )

    x_seq, y_seq = make_sequences_from_frames(
        frames_arr,
        labels_arr,
        seq_len=seq_len,
        stride=stride,
        label_mode=label_mode,
        dtype_str=dtype_str,
    )

    np.save(x_out, x_seq)
    np.save(y_out, y_seq)

    print(f"  Saved X_seq to: {x_out}")
    print(f"  Saved y_seq to: {y_out}")

    return {
        "session_id": session_id,
        "view": view,
        "frames": int(frames_arr.shape[0]),
        "seq_samples": int(x_seq.shape[0]),
        "pos_frames": int(np.sum(labels_arr == 1)),
        "pos_sequences": int(np.sum(y_seq == 1)),
        "skipped": 0,
    }


# ======================
# Main
# ======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument(
        "--label_mode",
        type=str,
        default=DEFAULT_LABEL_MODE,
        choices=["any", "max", "center", "majority"],
    )
    ap.add_argument("--raw_video_dir", type=str, default=str(DEFAULT_RAW_VIDEO_DIR))
    ap.add_argument("--excel_path", type=str, default=str(DEFAULT_EXCEL_PATH))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--dtype", type=str, default=DEFAULT_DTYPE_STR, choices=["uint8", "float32"])
    args = ap.parse_args()

    seq_len = int(args.seq_len)
    stride = int(args.stride)
    label_mode = args.label_mode
    dtype_str = args.dtype
    raw_video_dir = Path(args.raw_video_dir).expanduser()
    excel_path = Path(args.excel_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_sessions.json"

    if not raw_video_dir.is_dir():
        raise FileNotFoundError(f"RAW_VIDEO_DIR not found: {raw_video_dir}")

    print(f"Using RAW_VIDEO_DIR: {raw_video_dir}")

    if not excel_path.exists():
        print(f"Warning: Excel file not found: {excel_path}")
    else:
        print(f"Using Excel file: {excel_path}")

    print(f"Using OUTPUT_DIR: {output_dir}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print(f"[CONFIG] seq_len={seq_len}, stride={stride}, label_mode={label_mode}, dtype={dtype_str}")

    video_sessions = scan_videos(raw_video_dir)
    excel_sessions = scan_excel_sessions(excel_path)
    common_sessions = print_session_match_summary(video_sessions, excel_sessions)

    if not common_sessions:
        print("[ERROR] No sessions with both video and Excel to process.")
        return

    sessions_env = os.environ.get("SESSIONS", "").strip()
    if sessions_env:
        requested = {s.strip() for s in sessions_env.split(",") if s.strip()}
        common_set = set(common_sessions)
        target_sessions = sorted(common_set.intersection(requested))
        missing = requested - common_set
        if missing:
            print("\n[WARN] These requested sessions are missing video and/or Excel:", missing)
        if not target_sessions:
            print("[ERROR] No valid sessions to process after filtering by SESSIONS env var.")
            return
    else:
        target_sessions = common_sessions

    print("\nSessions to process:")
    for sid in target_sessions:
        print(f"  - {sid}")

    manifest_entries: List[Dict[str, int]] = []

    for session_id in tqdm(target_sessions, desc="Sessions"):
        sheet_name = excel_sessions[session_id]
        intervals = load_seizure_intervals(excel_path, sheet_name)
        if not intervals:
            print(f"[WARN] No intervals found for session {session_id} (sheet {sheet_name}), skipping.")
            continue

        view_to_path = video_sessions.get(session_id, {})

        for view in VIEWS:
            if view not in view_to_path:
                print(f"[WARN] Session {session_id} has no video for view {view}, skipping view.")
                continue

            stats = process_session_view(
                session_id=session_id,
                view=view,
                video_path=view_to_path[view],
                intervals=intervals,
                seq_len=seq_len,
                stride=stride,
                label_mode=label_mode,
                output_dir=output_dir,
                dtype_str=dtype_str,
            )
            manifest_entries.append(stats)

    if manifest_entries:
        try:
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            else:
                existing = []

            existing.extend(manifest_entries)
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

            print(f"\nSaved/updated manifest: {manifest_path}")
        except Exception as e:
            print(f"[WARN] Could not write manifest: {e}")

    print("\nSequence preprocessing complete.")


if __name__ == "__main__":
    main()
