#!/usr/bin/env python3
"""
Alignment Check for Multi-View Recordings
-----------------------------------------

This script analyzes temporal alignment between the three camera views
for each rat session.

CSV output is saved to:
    data/alignment_summary.csv

Assumptions:
    - Input directory: ~/gcs/inputs
    - Filenames like:
        POST KA061725 F1-webcamside1.mp4
        POST KA061725 F1-webcamside2.mp4
        POST KA061725 F1-webcamup.mp4

View mapping:
    webcamup     -> TOP
    webcamside1  -> SIDE1
    webcamside2  -> SIDE2

Requirements:
    - OpenCV (cv2)
    - NumPy

Author: Hyojin Seo
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("~/gcs/inputs").expanduser()
DEFAULT_STEP_FRAMES = 30
DEFAULT_OUTPUT_CSV = Path("~/seizure_detector/data/alignment_summary.csv")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    view: str
    fps: float
    n_frames: int
    duration_sec: float


# ---------------------------------------------------------------------------
# File scanning & grouping
# ---------------------------------------------------------------------------

def list_mp4_files(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]


def extract_session_id(filename: str) -> Optional[str]:
    match = re.search(r"(KA\d{6}\s+[FM]\d+)", filename)
    return match.group(1) if match else None


def infer_view_from_name(filename: str) -> str:
    name = filename.lower()
    if "webcamup" in name:
        return "TOP"
    if "webcamside1" in name:
        return "SIDE1"
    if "webcamside2" in name:
        return "SIDE2"
    return "UNKNOWN"


def get_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_sec = n_frames / fps if fps > 0 else 0.0
    cap.release()

    view = infer_view_from_name(path.name)
    return VideoInfo(path=path, view=view, fps=fps, n_frames=n_frames, duration_sec=duration_sec)


def group_videos_by_session(video_files: List[Path]) -> Dict[str, Dict[str, VideoInfo]]:
    sessions: Dict[str, Dict[str, VideoInfo]] = {}
    for path in video_files:
        session_id = extract_session_id(path.name)
        if session_id is None:
            continue
        info = get_video_info(path)
        session_dict = sessions.setdefault(session_id, {})
        session_dict[info.view] = info
    return sessions


# ---------------------------------------------------------------------------
# Brightness series & alignment estimation
# ---------------------------------------------------------------------------

def brightness_series(path: Path, step: int = DEFAULT_STEP_FRAMES) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for brightness: {path}")

    series = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            series.append(float(gray.mean()))
        frame_idx += 1

    cap.release()
    return np.array(series, dtype=np.float32)


def estimate_offset(ref_series: np.ndarray, other_series: np.ndarray, step: int, ref_fps: float) -> Tuple[int, float]:
    if ref_series.size == 0 or other_series.size == 0:
        return 0, 0.0

    length = min(ref_series.size, other_series.size)

    ref = ref_series[:length] - ref_series[:length].mean()
    other = other_series[:length] - other_series[:length].mean()

    corr = np.correlate(ref, other, mode="full")
    lags = np.arange(-length + 1, length)
    best_lag_sampled = int(lags[np.argmax(corr)])

    best_lag_frames = best_lag_sampled * step
    offset_sec = best_lag_frames / ref_fps if ref_fps > 0 else 0.0

    return best_lag_frames, offset_sec


# ---------------------------------------------------------------------------
# Tri-view visualization (optional)
# ---------------------------------------------------------------------------

def visualize_session_tri_view(session_id: str, session_videos: Dict[str, VideoInfo]) -> None:
    top = session_videos.get("TOP")
    s1 = session_videos.get("SIDE1")
    s2 = session_videos.get("SIDE2")

    if top is None:
        print(f"No TOP view for {session_id}")
        return

    caps = {
        "TOP": cv2.VideoCapture(str(top.path)),
        "SIDE1": cv2.VideoCapture(str(s1.path)) if s1 else None,
        "SIDE2": cv2.VideoCapture(str(s2.path)) if s2 else None,
    }

    try:
        while True:
            frames = {}
            for v, cap in caps.items():
                if cap is None:
                    frames[v] = None
                    continue
                ret, f = cap.read()
                frames[v] = f if ret else None

            if frames["TOP"] is None:
                break

            target_h = 360
            resized = []
            for v in ["TOP", "SIDE1", "SIDE2"]:
                f = frames.get(v)
                if f is None:
                    continue
                h, w = f.shape[:2]
                resized.append(cv2.resize(f, (int(w * target_h / h), target_h)))

            combined = np.hstack(resized)
            cv2.imshow(f"{session_id} | TOP | SIDE1 | SIDE2", combined)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        for cap in caps.values():
            if cap:
                cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main alignment analysis
# ---------------------------------------------------------------------------

def analyze_alignment(input_dir: Path, step: int, output_csv: Path) -> None:
    print(f"Scanning directory: {input_dir}")

    video_files = list_mp4_files(input_dir)
    if not video_files:
        print("No .mp4 files found.")
        return

    sessions = group_videos_by_session(video_files)
    print(f"Found {len(sessions)} sessions.")

    # Ensure data/ folder exists
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "view", "path", "fps", "n_frames",
            "duration_sec", "offset_frames_vs_TOP", "offset_sec_vs_TOP"
        ])

        for session_id, views in sorted(sessions.items()):
            print("-" * 60)
            print(f"Session: {session_id}")

            top = views.get("TOP")
            if top is None:
                print("  Missing TOP view. Skipping offset calc.")
                for view_name, info in views.items():
                    writer.writerow([session_id, view_name, str(info.path),
                                     f"{info.fps:.6f}", info.n_frames,
                                     f"{info.duration_sec:.3f}",
                                     0, 0.0])
                continue

            # compute TOP brightness
            top_series = brightness_series(top.path, step)

            # write TOP row
            writer.writerow([
                session_id, "TOP", str(top.path),
                f"{top.fps:.6f}", top.n_frames, f"{top.duration_sec:.3f}",
                0, 0.0
            ])

            # for SIDE views
            for view_name in ["SIDE1", "SIDE2"]:
                info = views.get(view_name)
                if info is None:
                    continue

                other_series = brightness_series(info.path, step)

                lag_frames, lag_sec = estimate_offset(
                    ref_series=top_series,
                    other_series=other_series,
                    step=step,
                    ref_fps=top.fps
                )

                print(f"  Offset {view_name} vs TOP: {lag_frames} frames ({lag_sec:.3f} sec)")

                writer.writerow([
                    session_id, view_name, str(info.path),
                    f"{info.fps:.6f}", info.n_frames, f"{info.duration_sec:.3f}",
                    lag_frames, f"{lag_sec:.6f}"
                ])

    print(f"\nAlignment summary saved to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check temporal alignment of multi-view recordings.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_FRAMES)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--visualize", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_alignment(args.input_dir, args.step, args.output_csv)

    if args.visualize:
        video_files = list_mp4_files(args.input_dir)
        sessions = group_videos_by_session(video_files)
        session_videos = sessions.get(args.visualize)
        if session_videos:
            visualize_session_tri_view(args.visualize, session_videos)
        else:
            print(f"Session '{args.visualize}' not found.")


if __name__ == "__main__":
    main()
