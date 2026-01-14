#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combining script for sequence-based CNN + LSTM/GRU model.

This script collects all per-session npy files created by
    preprocess_seq_cnn_lstm.py

It produces:
    - X_SEQ_<VIEW>_combined.npy (per-view combined sequences)
    - y_SEQ_combined.npy        (combined sequence labels)
    - manifest_combined.json    (records which sessions were included)

Author: Hyojin Seo
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# ======================
# Configuration
# ======================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Per-session npy directory (created by preprocess_seq_cnn_lstm.py)
SESSION_DIR = PROJECT_ROOT / "data" / "processed_seq" / "sessions"

# Output directory for combined npys
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_seq" / "combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Manifest of combined results
MANIFEST_PATH = OUTPUT_DIR / "manifest_combined.json"

# Views used in this project
VIEWS = ["TOP", "SIDE", "SIDE2"]


# ======================
# Helper functions
# ======================

def scan_session_files(session_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Return a mapping:

        combined_sessions[session_id][view] = X_SEQ file path

    A session_id is the normalized identifier in filenames:
        e.g. "112625F1", "112625F1_B"

    File format examples:
        X_SEQ_TOP_112625F1.npy
        y_SEQ_112625F1.npy
    """
    sessions: Dict[str, Dict[str, Path]] = {}

    for fname in os.listdir(session_dir):
        if not fname.endswith(".npy"):
            continue

        p = session_dir / fname
        name = fname

        # Matching X_SEQ_TOP_112625F1.npy → view=TOP, sess=112625F1
        if name.startswith("X_SEQ_"):
            parts = name.split("_")  # ["X", "SEQ", "TOP", "112625F1.npy"]
            if len(parts) < 4:
                continue

            view = parts[2]
            if view not in VIEWS:
                continue

            sess_id = parts[3].replace(".npy", "")
            sess_entry = sessions.setdefault(sess_id, {})
            sess_entry[view] = p

        # Matching y_SEQ_112625F1.npy → label file
        elif name.startswith("y_SEQ_"):
            sess_id = name.replace("y_SEQ_", "").replace(".npy", "")
            sess_entry = sessions.setdefault(sess_id, {})
            sess_entry["LABEL"] = p

    return sessions


def combine_by_view(sessions: Dict[str, Dict[str, Path]], view: str, out_dir: Path) -> Path:
    """
    Combine all X_SEQ_<view>_<session>.npy into one file.

    Returns the output path.
    """
    view_arrays = []
    included_sessions = []

    for sess_id, files in sessions.items():
        if view not in files:
            continue
        arr = np.load(files[view])
        view_arrays.append(arr)
        included_sessions.append(sess_id)

    if not view_arrays:
        print(f"[WARN] No data found for view {view}.")
        return None

    combined = np.concatenate(view_arrays, axis=0)
    out_path = out_dir / f"X_SEQ_{view}_combined.npy"
    np.save(out_path, combined)

    print(f"Saved combined {view} → {out_path} | Shape: {combined.shape}")
    return out_path


def combine_labels(sessions: Dict[str, Dict[str, Path]], out_dir: Path) -> Path:
    """
    Combine all y_SEQ_<session>.npy into one label file.
    """
    label_arrays = []
    included_sessions = []

    for sess_id, files in sessions.items():
        if "LABEL" not in files:
            continue
        y = np.load(files["LABEL"])
        label_arrays.append(y)
        included_sessions.append(sess_id)

    if not label_arrays:
        print("[WARN] No label files found.")
        return None

    combined = np.concatenate(label_arrays, axis=0)
    out_path = out_dir / "y_SEQ_combined.npy"
    np.save(out_path, combined)

    print(f"Saved combined labels → {out_path} | Shape: {combined.shape}")
    return out_path


# ======================
# Main
# ======================

def main():
    """
    Main combining workflow:

        - Scan per-session npys
        - Combine X_SEQ_<view> for each view
        - Combine y_SEQ labels
        - Write manifest_combined.json
    """

    print(f"Scanning session directory: {SESSION_DIR}")

    sessions = scan_session_files(SESSION_DIR)
    if not sessions:
        print("[ERROR] No per-session npy files found. Run preprocessing first.")
        return

    print(f"Found {len(sessions)} session entries to combine.")

    manifest: Dict[str, List[str]] = {
        "sessions_found": sorted(sessions.keys()),
        "combined_views": [],
        "combined_labels": False,
    }

    # Combine per view
    for view in VIEWS:
        out_path = combine_by_view(sessions, view, OUTPUT_DIR)
        if out_path is not None:
            manifest["combined_views"].append(view)

    # Combine labels
    y_out = combine_labels(sessions, OUTPUT_DIR)
    if y_out is not None:
        manifest["combined_labels"] = True

    # Save manifest_combined.json
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest: {MANIFEST_PATH}")
    print("Combination complete.")


if __name__ == "__main__":
    main()
