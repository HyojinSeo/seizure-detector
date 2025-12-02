#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing script for seizure-detector project.

- Reads multi-view videos (TOP, SIDE, SIDE2) from ~/gcs/inputs
- Matches sessions to Excel sheets (oct_seizure_stage_filtered.xlsx)
- Extracts frames at 1 fps, grayscale, resized to 128x128
- Builds per-session npy files for each view and per-session labels
- Combines all sessions into view-wise combined npy files and a combined label array

Author: Hyojin Seo
"""

import os
import re
from collections import defaultdict
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

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video input folder (mount point from GCS)
DATA_DIR = Path("~/gcs/inputs").expanduser()

# Output folder for npy files
OUTPUT_DIR = PROJECT_ROOT / "data" / "npy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Excel with seizure labels
EXCEL_PATH = PROJECT_ROOT / "data" / "oct_seizure_stage_filtered.xlsx"

# Views to use
VIEWS = ["TOP", "SIDE", "SIDE2"]

# File prefixes
NPY_PREFIX = "X_"
LABEL_PREFIX = "y_"

# ======================
# Sanity check for input directory and Excel
# ======================

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
print(f"Using DATA_DIR: {DATA_DIR}")
print(f"Found {len(list(DATA_DIR.iterdir()))} files in DATA_DIR")

if not EXCEL_PATH.exists():
    print(f"Warning: Excel file not found: {EXCEL_PATH}")
else:
    print(f"Using Excel file: {EXCEL_PATH}")

# ======================
# Sheet name normalization for matching
# ======================

sheet_list = []
if EXCEL_PATH.exists():
    try:
        sheet_list = pd.ExcelFile(EXCEL_PATH).sheet_names
    except Exception as e:
        print(f"Warning: Failed to read Excel sheets from {EXCEL_PATH}: {e}")
        sheet_list = []

def normalize_key(s: str) -> str:
    """Uppercase; remove spaces and hyphens for robust matching."""
    return s.upper().replace("-", "").replace(" ", "")

normalized_sheets = {normalize_key(s): s for s in sheet_list}


# ======================
# Helper functions
# ======================

def infer_view_from_name(filename_lower: str) -> str:
    """
    Infer view from filename tokens.

    Rules:
        webcamup     / top   -> TOP
        webcamside1  / side  -> SIDE
        webcamside2  / side2 -> SIDE2
    """
    name = filename_lower
    if "webcamup" in name or name.endswith("top.mp4") or "top" in name:
        return "TOP"
    if "webcamside2" in name or name.endswith("side2.mp4") or "side2" in name:
        return "SIDE2"
    if "webcamside1" in name or name.endswith("side.mp4") or "side" in name:
        return "SIDE"
    return "UNKNOWN"


def extract_key_from_filename(filename: str) -> str:
    """
    Derive session key in the exact sheet naming form: 'KA###### Xn'

    Example:
        'POST KA061725 F1-webcamup.mp4' -> 'KA061725 F1'
    """
    name = os.path.splitext(filename)[0]

    # Remove common view tokens
    for token in ["webcamup", "webcamside1", "webcamside2", "top", "side2", "side"]:
        name = re.sub(token, " ", name, flags=re.IGNORECASE)

    # Replace separators by spaces
    name = re.sub(r"[-_\.]+", " ", name).strip()
    parts = name.split()

    # Find a token like KA061725
    ka_idx = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"(?i)KA\d+", p):
            ka_idx = i
            break
        if p.upper().startswith("KA") and p[2:].isdigit():
            ka_idx = i
            break

    # Find next token like F1 or M2
    def is_subject_token(tok: str) -> bool:
        return re.fullmatch(r"[FM]\d+", tok.upper()) is not None

    subj = None
    if ka_idx is not None and ka_idx + 1 < len(parts):
        if is_subject_token(parts[ka_idx + 1]):
            subj = parts[ka_idx + 1].upper()

    if ka_idx is not None:
        ka = parts[ka_idx].upper()
        if not ka.startswith("KA"):
            ka = "KA" + ka
        if subj:
            return f"{ka} {subj}"
        return ka

    # Fallback: normalized whole name
    return " ".join(parts).upper()


def get_seizure_frame_ranges(sheet_name: str, fps: int = FPS_TARGET):
    """
    Read (start, end, label) from the specified sheet.
    Units in Excel are assumed to be seconds.
    Only rows with label == 0 are seizures.
    Returns a list of (start_frame, end_frame) at the given fps.
    """
    if not EXCEL_PATH.exists() or not sheet_name:
        return []

    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
    except Exception as e:
        print(f"Warning: Failed to read sheet '{sheet_name}': {e}")
        return []

    cols = {c.lower(): c for c in df.columns}
    required = ["start", "end", "label"]
    if not all(k in cols for k in required):
        print(f"Warning: Sheet '{sheet_name}' does not have required columns {required}")
        return []

    start_col, end_col, label_col = cols["start"], cols["end"], cols["label"]
    ranges = []
    for _, row in df.iterrows():
        try:
            s = int(row[start_col])
            e = int(row[end_col])
            lab = int(row[label_col])
            if s > e:
                s, e = e, s
            s_f, e_f = int(round(s * fps)), int(round(e * fps))
            # Here we assume label == 0 means seizure
            if lab == 0:
                ranges.append((s_f, e_f))
        except Exception:
            continue
    return ranges


def extract_frames(video_path: Path, seizure_ranges=None,
                   resize=(128, 128), fps: int = FPS_TARGET):
    """
    Extract frames from video at given fps, grayscale normalized to [0, 1].

    Returns:
        X_arr: (num_frames, H, W, 1)
        y_arr: (num_frames,) if seizure_ranges is given, else None
    """
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened() or video_fps == 0:
        print(f"Warning: Could not open video: {video_path}")
        cap.release()
        return np.array([]), None

    frame_interval = max(int(round(video_fps / fps)), 1)
    count, saved = 0, 0
    X_data, y_data = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, resize)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray = np.expand_dims(gray, axis=-1)
            X_data.append(gray)

            if seizure_ranges is not None:
                in_seizure = any(s <= saved <= e for s, e in seizure_ranges)
                y_data.append(1 if in_seizure else 0)

            saved += 1
        count += 1

    cap.release()
    X_arr = np.array(X_data, dtype=DTYPE)
    if seizure_ranges is not None:
        y_arr = np.array(y_data, dtype=np.uint8)
        return X_arr, y_arr
    return X_arr, None


# ======================
# Step 1: Group videos by session
# ======================

video_groups: dict[str, dict[str, Path]] = defaultdict(dict)

for path in DATA_DIR.iterdir():
    if not path.is_file():
        continue
    if path.name.startswith("._") or path.suffix.lower() != ".mp4":
        continue

    view = infer_view_from_name(path.name.lower())
    if view == "UNKNOWN":
        continue

    key = extract_key_from_filename(path.name)  # like 'KA061725 F1'
    video_groups[key][view] = path

print(f"\nFound {len(video_groups)} session candidates.\n")

# ======================
# Step 2: Check session-to-sheet matches
# ======================

print("Checking session-to-sheet matches:\n")
for key in sorted(video_groups.keys()):
    norm_key = normalize_key(key)
    if norm_key in normalized_sheets:
        print(f"Matched:     {key.ljust(20)} --> Sheet: {normalized_sheets[norm_key]}")
    else:
        print(f"NOT matched: {key}")
        print("  -> No sheet with this name in Excel. This session will be SKIPPED.")

# ======================
# Step 3: Process and save per-session npy files
# ======================

print("\nStarting preprocessing (per session)...\n")

for key, views_dict in tqdm(video_groups.items()):
    # Require all three views for now
    if not all(v in views_dict for v in VIEWS):
        print(f"[SKIP] Missing view(s) for {key}")
        continue

    norm_key = normalize_key(key)
    if norm_key not in normalized_sheets:
        print(f"[SKIP] No matching sheet for {key}")
        continue

    sheet_name = normalized_sheets[norm_key]
    seizure_ranges = get_seizure_frame_ranges(sheet_name=sheet_name, fps=FPS_TARGET)

    data = {}
    y = None
    min_len = float("inf")

    # Extract frames for each view
    for v in VIEWS:
        X_v, y_v = extract_frames(
            video_path=views_dict[v],
            seizure_ranges=seizure_ranges,
            resize=RESIZE_SHAPE,
            fps=FPS_TARGET,
        )
        data[v] = X_v

        if X_v.shape[0] < min_len:
            min_len = X_v.shape[0]

        if y_v is not None and y is None:
            y = y_v

    if min_len == float("inf") or min_len == 0:
        print(f"[SKIP] No frames extracted for {key}")
        continue

    # Trim all views and labels to the same length
    for v in VIEWS:
        data[v] = data[v][:min_len]
        out_path = OUTPUT_DIR / f"{NPY_PREFIX}{v}_{key}_float32.npy"
        np.save(out_path, data[v].astype(DTYPE))

    if y is not None:
        y = y[:min_len].astype(np.uint8)
        y_path = OUTPUT_DIR / f"{LABEL_PREFIX}{key}.npy"
        np.save(y_path, y)

# ======================
# Step 4: Combine per view into single npy
# ======================

print("\nCombining all sessions into final .npy files...\n")

for v in VIEWS:
    print(f"\nCombining view: {v}")
    files = sorted(
        [
            OUTPUT_DIR / f
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith(f"{NPY_PREFIX}{v}_") and f.endswith("_float32.npy")
        ]
    )
    if not files:
        print(f"No files for view {v}")
        continue

    total_frames, shape_sample = 0, None
    valid_files = []

    # First pass: check shapes and total frame count
    for path in files:
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr)
        if shape_sample is None:
            shape_sample = arr.shape[1:]
        if arr.shape[1:] == shape_sample:
            total_frames += arr.shape[0]
            valid_files.append(path)
        else:
            print(f"Shape mismatch: {path} (got {arr.shape[1:]}, expect {shape_sample})")

    if shape_sample is None or total_frames == 0:
        print(f"No valid frames for view {v}")
        continue

    combined = np.empty((total_frames, *shape_sample), dtype=DTYPE)
    start = 0
    for path in valid_files:
        arr = np.load(path)
        end = start + arr.shape[0]
        combined[start:end] = arr
        start = end

    out_path = OUTPUT_DIR / f"{NPY_PREFIX}{v}_combined.npy"
    np.save(out_path, combined)
    print(f"Saved: {out_path} | Shape: {combined.shape}")

# ======================
# Step 5: Combine labels
# ======================

label_files = sorted(
    [
        OUTPUT_DIR / f
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith(LABEL_PREFIX) and f.endswith(".npy")
    ]
)

if label_files:
    y_all = [np.load(f) for f in label_files]
    y_combined = np.concatenate(y_all).astype(np.uint8)
    y_path = OUTPUT_DIR / "y_combined.npy"
    np.save(y_path, y_combined)
    print(f"\nSaved combined labels to: {y_path} | Shape: {y_combined.shape}")
else:
    print("\nNo labels found to combine.")

print("\nPreprocessing and combining complete.")
