#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence preprocessing script for seizure-detector project.

End-to-end pipeline for the CNN + LSTM/GRU model:

- Reads multi-view POST KA videos (TOP, SIDE, SIDE2) from ~/gcs/inputs
- Matches sessions to Excel sheets (seizure_stage.xlsx)
- Extracts frames at 1 fps, grayscale, resized to 128x128
- Labels each frame as seizure/non-seizure from Excel time intervals
- Builds sliding-window sequences (T frames) per session and per view
- Saves per-session npy files for each view (X_SEQ_*, y_SEQ_*)

Author: Hyojin Seo
"""

import os
import re
import json
from typing import Dict, List, Tuple

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ======================
# Configuration
# ======================

# Frame sampling parameters
FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)
DTYPE = np.float32

# Sequence parameters
SEQ_LEN = 16
STRIDE = 4
LABEL_MODE = "any"  # "any", "max", "center", "majority"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video input folder (mount point from GCS)
RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()

# Excel with seizure intervals
EXCEL_PATH = PROJECT_ROOT / "data" / "seizure_stage.xlsx"

# Output folder for per-session sequence npy files
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_seq" / "sessions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Views to use and matching substrings in filenames
VIEWS = ["TOP", "SIDE", "SIDE2"]
VIEW_CONFIG = {
    "TOP": "webcamup",
    "SIDE": "webcamside1",
    "SIDE2": "webcamside2",
}

# Allowed video extensions
VALID_EXTS = (".mp4",)

# Manifest file to keep track of processed sessions
MANIFEST_PATH = OUTPUT_DIR / "manifest_sessions.json"


# ======================
# Sanity checks
# ======================

if not RAW_VIDEO_DIR.is_dir():
    raise FileNotFoundError(f"RAW_VIDEO_DIR not found: {RAW_VIDEO_DIR}")
print(f"Using RAW_VIDEO_DIR: {RAW_VIDEO_DIR}")

if not EXCEL_PATH.exists():
    print(f"Warning: Excel file not found: {EXCEL_PATH}")
else:
    print(f"Using Excel file: {EXCEL_PATH}")


# ======================
# Helper functions: sheet/session naming
# ======================

def get_session_id_from_sheet(sheet_name: str) -> str:
    """
    Convert sheet name like '112625F1' or '112625F1_B' into a session id string:

        '112625F1'   -> '112625 F1'
        '112625F1_B' -> '112625 F1_B'

    The internal session id is used for matching and logging.
    """
    clean = sheet_name.replace("-", "").replace(" ", "")
    m = re.match(r"^(\d{6})([MF]\d)(?:_B)?$", clean, re.IGNORECASE)
    if not m:
        return None
    date = m.group(1)
    rat = m.group(2).upper()
    is_booster = clean.upper().endswith("_B")
    if is_booster:
        return f"{date} {rat}_B"
    return f"{date} {rat}"


def normalize_session_for_filename(session_id: str) -> str:
    """
    Normalize session id like '112625 F1_B' to '112625F1_B' for filenames.
    """
    return session_id.replace(" ", "")


# ======================
# Helper functions: time parsing and intervals
# ======================

def parse_hms_to_seconds(s: str) -> float:
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
        elif len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return h * 3600 + m * 60 + sec
        else:
            return None
    except ValueError:
        return None


def parse_time_interval(interval_str: str) -> Tuple[float, float]:
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
    for (start, end) in intervals:
        if t_sec >= start and t_sec <= end:
            return 1
    return 0


# ======================
# Helper functions: video and Excel scanning
# ======================

def scan_videos(raw_video_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan raw_video_dir for POST KA .mp4 videos and return a mapping:

        sessions[session_id][view] = video_path

    session_id example: "112625 F1", "112625 F1_B"
    view is one of "TOP", "SIDE", "SIDE2".
    """
    sessions: Dict[str, Dict[str, Path]] = {}

    # Pattern: POST KA112625 F1 or POST KA112625 F1 B
    pattern = re.compile(
        r"POST\s*KA(\d{6})\s*([MF]\d)(?:\s*B)?",
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
        is_booster = " b-" in lower or " b." in lower or " b " in lower
        if is_booster:
            session_id = f"{date} {rat}_B"
        else:
            session_id = f"{date} {rat}"

        # Determine view from substring
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

    Returns
    -------
    session_to_sheet : dict
        Keys are session ids like "112625 F1" or "112625 F1_B",
        values are original sheet names like "112625F1_B".
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


def print_session_match_summary(video_sessions: Dict[str, Dict[str, Path]],
                                excel_sessions: Dict[str, str]) -> List[str]:
    """
    Print a summary showing which sessions have both videos and Excel sheets.

    Returns
    -------
    common_sessions : list of session ids that have both video and Excel.
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
    X_frames: np.ndarray,
    y_frames: np.ndarray,
    seq_len: int,
    stride: int,
    label_mode: str = "any",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from frame-level arrays.

    X_frames : (N_frames, H, W) or (N_frames, H, W, C)
    y_frames : (N_frames,) or (N_frames, 1) with 0/1 labels.

    Returns:
        X_seq : (N_seq, T, H, W, C)
        y_seq : (N_seq,)
    """
    X = X_frames
    y = y_frames

    if y.ndim > 1:
        y = y.reshape(-1)

    n_frames = X.shape[0]
    if n_frames != y.shape[0]:
        raise ValueError(f"X and y have different number of frames: {n_frames} vs {y.shape[0]}")

    if X.ndim == 3:
        # (N, H, W) -> (N, H, W, 1)
        X = X[..., np.newaxis]

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

    X_seq = np.stack(sequences, axis=0).astype(DTYPE)
    y_seq = np.array(labels, dtype=np.int64)

    print(
        f"Built {X_seq.shape[0]} sequences of length {seq_len} "
        f"from {n_frames} frames (stride={stride}). "
        f"Positive sequences: {np.sum(y_seq == 1)} / {y_seq.shape[0]}"
    )

    return X_seq, y_seq


def process_session_view(
    session_id: str,
    view: str,
    video_path: Path,
    intervals: List[Tuple[float, float]],
    seq_len: int,
    stride: int,
    label_mode: str,
) -> Dict[str, int]:
    """
    For a given session and view:

    - Open the video.
    - Sample frames at 1 fps.
    - Convert to grayscale and resize to 128x128.
    - Label each frame based on seizure intervals.
    - Build sequences and save X_SEQ and y_SEQ npy files.

    Returns a small dict with counts for manifest logging.
    """
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
        }

    frames_arr = np.stack(frames, axis=0)  # (N_frames, H, W)
    labels_arr = np.array(labels, dtype=np.int64)  # (N_frames,)

    print(
        f"  Collected {frames_arr.shape[0]} frames. "
        f"Positive frames: {np.sum(labels_arr == 1)}"
    )

    X_seq, y_seq = make_sequences_from_frames(
        frames_arr,
        labels_arr,
        seq_len=seq_len,
        stride=stride,
        label_mode=label_mode,
    )

    norm_sess = normalize_session_for_filename(session_id)

    x_out = OUTPUT_DIR / f"X_SEQ_{view}_{norm_sess}.npy"
    y_out = OUTPUT_DIR / f"y_SEQ_{view}_{norm_sess}.npy"

    np.save(x_out, X_seq)
    np.save(y_out, y_seq)

    print(f"  Saved X_seq to: {x_out}")
    print(f"  Saved y_seq to: {y_out}")

    return {
        "session_id": session_id,
        "view": view,
        "frames": int(frames_arr.shape[0]),
        "seq_samples": int(X_seq.shape[0]),
        "pos_frames": int(np.sum(labels_arr == 1)),
        "pos_sequences": int(np.sum(y_seq == 1)),
    }


# ======================
# Main
# ======================

def main():
    """
    Main entry point.

    Workflow:
        - Scan videos and Excel for sessions
        - Print match summary
        - Optionally filter to a subset of sessions via SESSIONS env var
        - For each session+view, build sequence npy files
        - Append results to manifest_sessions.json
    """
    # Scan videos and Excel
    video_sessions = scan_videos(RAW_VIDEO_DIR)
    excel_sessions = scan_excel_sessions(EXCEL_PATH)
    common_sessions = print_session_match_summary(video_sessions, excel_sessions)

    if not common_sessions:
        print("[ERROR] No sessions with both video and Excel to process.")
        return

    # Optional: environment variable to restrict sessions (simple mechanism)
    # Example: export SESSIONS="112625 F1,112625 F1_B"
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

    # Process each session
    for session_id in tqdm(target_sessions, desc="Sessions"):
        sheet_name = excel_sessions[session_id]
        intervals = load_seizure_intervals(EXCEL_PATH, sheet_name)
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
                seq_len=SEQ_LEN,
                stride=STRIDE,
                label_mode=LABEL_MODE,
            )
            manifest_entries.append(stats)

    # Update manifest
    if manifest_entries:
        try:
            if MANIFEST_PATH.exists():
                with MANIFEST_PATH.open("r") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            else:
                existing = []

            existing.extend(manifest_entries)
            with MANIFEST_PATH.open("w") as f:
                json.dump(existing, f, indent=2)

            print(f"\nSaved/updated manifest: {MANIFEST_PATH}")
        except Exception as e:
            print(f"[WARN] Could not write manifest: {e}")

    print("\nSequence preprocessing complete.")


if __name__ == "__main__":
    main()
