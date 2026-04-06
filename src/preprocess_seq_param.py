#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parametric sequence preprocessing for seizure-detector.

- Reads multi-view POST KA videos from ~/gcs/inputs
- Matches sessions to Excel sheets (data/seizure_stage.xlsx)
- Extracts 1 fps grayscale frames resized to 128x128
- Labels each frame by Excel intervals
- Builds sliding-window sequences (seq_len, stride)
- Saves per-session npy for each view into a configurable sessions_dir

Important:
- To avoid overwriting different (seq_len/stride) settings, you MUST write to different sessions_dir.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)
DTYPE = np.float32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()
EXCEL_PATH = PROJECT_ROOT / "data" / "seizure_stage.xlsx"

VIEWS = ["TOP", "SIDE", "SIDE2"]
VIEW_CONFIG = {
    "TOP": "webcamup",
    "SIDE": "webcamside1",
    "SIDE2": "webcamside2",
}
VALID_EXTS = (".mp4",)


def get_session_id_from_sheet(sheet_name: str) -> str:
    if not isinstance(sheet_name, str) or not sheet_name.strip():
        return None

    clean = sheet_name.replace(" ", "")
    m = re.match(r"^(\d{6})([MF]\d)(?:_B(?:[-_]?(\d+))?)?$", clean, re.IGNORECASE)
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
    return session_id.replace(" ", "")


def parse_hms_to_seconds(s: str) -> float:
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


def parse_time_interval(interval_str: str) -> Tuple[float, float]:
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
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "Time" not in df.columns:
        print(f"[WARN] Sheet {sheet_name} has no 'Time' column.")
        return []

    intervals: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        start_sec, end_sec = parse_time_interval(row["Time"])
        if start_sec is None or end_sec is None:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        intervals.append((start_sec, end_sec))
    return intervals


def label_time_by_intervals(t_sec: float, intervals: List[Tuple[float, float]]) -> int:
    for (start, end) in intervals:
        if start <= t_sec <= end:
            return 1
    return 0


def scan_videos(raw_video_dir: Path) -> Dict[str, Dict[str, Path]]:
    sessions: Dict[str, Dict[str, Path]] = {}
    pattern = re.compile(r"POST\s*KA(\d{6})\s*([MF]\d)", re.IGNORECASE)

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
            continue

        sessions.setdefault(session_id, {})[view] = path

    return sessions


def scan_excel_sessions(excel_path: Path) -> Dict[str, str]:
    session_to_sheet: Dict[str, str] = {}
    if not excel_path.exists():
        return session_to_sheet

    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        sid = get_session_id_from_sheet(sheet)
        if sid:
            session_to_sheet[sid] = sheet
    return session_to_sheet


def make_sequences_from_frames(X_frames: np.ndarray, y_frames: np.ndarray, seq_len: int, stride: int, label_mode: str):
    if y_frames.ndim > 1:
        y_frames = y_frames.reshape(-1)
    if X_frames.ndim == 3:
        X_frames = X_frames[..., np.newaxis]  # (N,H,W,1)

    n_frames = int(X_frames.shape[0])
    if n_frames != int(y_frames.shape[0]):
        raise ValueError(f"X/y frame mismatch: {n_frames} vs {y_frames.shape[0]}")
    if n_frames < seq_len:
        raise ValueError(f"Not enough frames: {n_frames} < seq_len={seq_len}")

    seqs = []
    labs = []
    for start in range(0, n_frames - seq_len + 1, stride):
        end = start + seq_len
        x_seq = X_frames[start:end]
        y_seq = y_frames[start:end]

        if label_mode == "any":
            lab = int(np.any(y_seq > 0))
        elif label_mode == "center":
            lab = int(y_frames[start + seq_len // 2])
        elif label_mode == "majority":
            lab = int(np.sum(y_seq > 0) > (seq_len / 2.0))
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        seqs.append(x_seq)
        labs.append(lab)

    X_seq = np.stack(seqs, axis=0).astype(DTYPE)
    y_seq = np.array(labs, dtype=np.int64)
    return X_seq, y_seq


def outputs_exist(output_dir: Path, session_id: str, view: str) -> bool:
    norm = normalize_session_for_filename(session_id)
    return (output_dir / f"X_SEQ_{view}_{norm}.npy").exists() and (output_dir / f"y_SEQ_{view}_{norm}.npy").exists()


def process_one_view(output_dir: Path, session_id: str, view: str, video_path: Path, intervals, seq_len, stride, label_mode, skip_existing: bool):
    norm = normalize_session_for_filename(session_id)
    x_out = output_dir / f"X_SEQ_{view}_{norm}.npy"
    y_out = output_dir / f"y_SEQ_{view}_{norm}.npy"

    if skip_existing and x_out.exists() and y_out.exists():
        print(f"[SKIP] {session_id} {view} (exists)")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0
    if duration_sec <= 0:
        cap.release()
        raise RuntimeError(f"Bad duration: {video_path} fps={native_fps} frames={frame_count}")

    frames = []
    labels = []
    t = 0.0
    while t < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, RESIZE_SHAPE)
        frames.append(gray)
        labels.append(label_time_by_intervals(t, intervals))
        t += 1.0 / FPS_TARGET

    cap.release()
    if not frames:
        print(f"[WARN] No frames: {session_id} {view}")
        return

    X_frames = np.stack(frames, axis=0)
    y_frames = np.array(labels, dtype=np.int64)

    X_seq, y_seq = make_sequences_from_frames(X_frames, y_frames, seq_len=seq_len, stride=stride, label_mode=label_mode)

    np.save(x_out, X_seq)
    np.save(y_out, y_seq)

    print(f"[OK] {session_id} {view} -> {x_out.name}  N={X_seq.shape[0]}  pos={int(np.sum(y_seq==1))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--label_mode", type=str, default="any", choices=["any", "center", "majority"])
    ap.add_argument("--sessions_dir", type=str, required=True, help="Output dir for per-session npy (e.g., data/processed_seq/sessions_T4_S2)")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--exclude_prefix", type=str, default="010626", help="Exclude sessions starting with this date (default excludes test). Use '' to disable.")
    ap.add_argument("--manifest_name", type=str, default="manifest_sessions.json")
    args = ap.parse_args()

    if not RAW_VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"RAW_VIDEO_DIR not found: {RAW_VIDEO_DIR}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")

    out_dir = Path(args.sessions_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / args.manifest_name

    video_sessions = scan_videos(RAW_VIDEO_DIR)
    excel_sessions = scan_excel_sessions(EXCEL_PATH)

    common = sorted(set(video_sessions.keys()).intersection(set(excel_sessions.keys())))

    if args.exclude_prefix is not None and args.exclude_prefix != "":
        common = [s for s in common if not s.startswith(args.exclude_prefix)]

    if not common:
        print("[ERROR] No sessions to process after filtering.")
        return

    print(f"[INFO] seq_len={args.seq_len} stride={args.stride} label_mode={args.label_mode}")
    print(f"[INFO] sessions_dir={out_dir}")
    print(f"[INFO] skip_existing={args.skip_existing}")
    print(f"[INFO] exclude_prefix={args.exclude_prefix}")

    entries = []
    for sid in tqdm(common, desc="Sessions"):
        sheet = excel_sessions[sid]
        intervals = load_seizure_intervals(EXCEL_PATH, sheet)
        if not intervals:
            print(f"[WARN] No intervals: {sid} sheet={sheet}")
            continue

        view_map = video_sessions.get(sid, {})
        for view in VIEWS:
            if view not in view_map:
                print(f"[WARN] Missing view {view} for {sid}")
                continue
            if args.skip_existing and outputs_exist(out_dir, sid, view):
                continue
            process_one_view(out_dir, sid, view, view_map[view], intervals, args.seq_len, args.stride, args.label_mode, args.skip_existing)
        entries.append({"session_id": sid, "sheet": sheet})

    # manifest
    try:
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(old, list):
                old = []
        else:
            old = []
        old.extend(entries)
        manifest_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"[OK] manifest -> {manifest_path}")
    except Exception as e:
        print(f"[WARN] manifest write failed: {e}")

    print("[OK] Preprocess done.")

if __name__ == "__main__":
    main()
