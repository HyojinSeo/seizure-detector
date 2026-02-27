#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess ONLY frames (1fps) + per-second labels, per session/view.

Output:
  data/processed_frames/sessions/
    X_FRAMES_TOP_<TOKEN>.npy     uint8  (N,H,W,1)  0..255
    X_FRAMES_SIDE_<TOKEN>.npy    uint8
    X_FRAMES_SIDE2_<TOKEN>.npy   uint8
    y_FRAMES_<TOKEN>.npy         uint8  (N,)  0/1
  data/processed_frames/sessions/meta_<TOKEN>.json
  data/processed_frames/sessions/manifest_frames.json (append-only)

Token examples:
  010626F1, 010626F1_B, 121325M3_B-1

Notes:
- Stores X as uint8 to save disk.
- y is per-second label aligned to 1fps frames.
- Later: training will window dynamically with any SEQ_LEN/STRIDE.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------
# Config
# ----------------------
FPS_TARGET = 1
RESIZE_SHAPE = (128, 128)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = Path("~/gcs/inputs").expanduser()
EXCEL_PATH = PROJECT_ROOT / "data" / "seizure_stage.xlsx"

OUT_DIR = PROJECT_ROOT / "data" / "processed_frames" / "sessions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = ["TOP", "SIDE", "SIDE2"]
VIEW_SUBSTR = {
    "TOP": "webcamup",
    "SIDE": "webcamside1",
    "SIDE2": "webcamside2",
}
VALID_EXTS = (".mp4",)

MANIFEST_PATH = OUT_DIR / "manifest_frames.json"

SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1").strip() in {"1", "true", "True", "YES", "yes"}
SESSIONS_ENV = os.environ.get("SESSIONS", "").strip()  # optional filter: "010626 F1,010626 F1 B"


# ----------------------
# Helpers: time parsing
# ----------------------
def hms_to_sec(s: str) -> Optional[float]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m = int(parts[0]); sec = float(parts[1])
            return 60*m + sec
        if len(parts) == 3:
            h = int(parts[0]); m = int(parts[1]); sec = float(parts[2])
            return 3600*h + 60*m + sec
    except Exception:
        return None
    return None


def parse_time_interval(cell: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(cell, str):
        return None, None
    if "-" not in cell:
        return None, None
    a, b = cell.split("-", 1)
    s = hms_to_sec(a.strip())
    e = hms_to_sec(b.strip())
    if s is None or e is None:
        return None, None
    if e < s:
        s, e = e, s
    return float(s), float(e)


def load_intervals(excel_path: Path, sheet: str) -> List[Tuple[float, float]]:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if "Time" not in df.columns:
        return []
    out: List[Tuple[float, float]] = []
    for cell in df["Time"].dropna().tolist():
        s, e = parse_time_interval(cell)
        if s is None or e is None:
            continue
        out.append((s, e))
    return out


def label_at_t(t_sec: float, intervals: List[Tuple[float, float]]) -> int:
    for s, e in intervals:
        if s <= t_sec <= e:
            return 1
    return 0


# ----------------------
# Helpers: session naming
# ----------------------
def sheet_to_session_id(sheet_name: str) -> Optional[str]:
    """
    '010626F1' -> '010626 F1'
    '010626F1_B' -> '010626 F1_B'
    '121325M3_B-1' -> '121325 M3_B-1'
    """
    if not isinstance(sheet_name, str):
        return None
    clean = sheet_name.replace(" ", "")
    m = re.match(r"^(\d{6})([MF]\d)(?:_B(?:[-_]?(\d+))?)?$", clean, re.IGNORECASE)
    if not m:
        return None
    date = m.group(1)
    animal = m.group(2).upper()
    bnum = m.group(3)

    if bnum:
        return f"{date} {animal}_B-{bnum}"
    if "_B" in clean.upper():
        return f"{date} {animal}_B"
    return f"{date} {animal}"


def normalize_token(session_id: str) -> str:
    # '010626 F1_B' -> '010626F1_B', keep '-'
    return session_id.replace(" ", "")


def normalize_session_input(s: str) -> str:
    """
    Accept:
      '010626 F1'
      '010626 F1 B'
      'KA010626 F1 B'
    Convert to session_id used by excel mapping:
      '010626 F1' / '010626 F1_B' / '010626 F1_B-1'
    """
    s = s.strip().upper().replace("KA", "")
    parts = s.split()
    if len(parts) < 2:
        raise ValueError(f"Bad session format: {s}")
    date = parts[0]
    animal = parts[1]
    if len(parts) >= 3 and parts[2].startswith("B"):
        b = parts[2].replace("_", "-")
        # "B" or "B-1"
        if b == "B":
            return f"{date} {animal}_B"
        return f"{date} {animal}_B-{b.split('-', 1)[1]}"
    return f"{date} {animal}"


# ----------------------
# Video matching
# ----------------------
def find_view_files(raw_dir: Path, session_id: str) -> Dict[str, Path]:
    """
    session_id examples:
      '010626 F1'
      '010626 F1_B'
      '121325 M3_B-1'
    Matches file names like:
      POST KA010626 F1 ... -webcamup.mp4
      POST KA010626 F1 B ... -webcamup.mp4
      POST KA121325 M3 B-1 ... -webcamup.mp4
    """
    token = normalize_token(session_id)  # e.g., 010626F1_B-1
    date = token[:6]
    # animal is second piece without date
    rest = token[6:]  # e.g., F1_B-1
    # animal like F1
    animal = rest.split("_", 1)[0]  # F1
    booster = "_B" in rest
    booster_tag = ""
    if booster:
        booster_tag = rest.split("_", 1)[1]  # "B" or "B-1"

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
            # must include "<animal> b"
            if f"{animal.lower()} b" not in name:
                return False
            if booster_tag and booster_tag != "B":
                # require b-1 or b 1
                num = booster_tag.split("-", 1)[1]
                if (f"b-{num}" not in name) and (f"b {num}" not in name) and (f"b_{num}" not in name):
                    return False
        else:
            # avoid booster
            if f"{animal.lower()} b" in name:
                return False

        return True

    out: Dict[str, Path] = {}
    for v in VIEWS:
        sub = VIEW_SUBSTR[v]
        cand = [p for p in raw_dir.iterdir() if p.is_file() and is_match(p, sub)]
        if len(cand) != 1:
            raise RuntimeError(f"Expected 1 match for {session_id} view={v}, found {len(cand)}: {[c.name for c in cand[:10]]}")
        out[v] = cand[0]
    return out


# ----------------------
# Frame extraction (uint8)
# ----------------------
def extract_frames_uint8(video_path: Path) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0
    if duration_sec <= 0:
        cap.release()
        raise RuntimeError(f"Bad duration: {video_path} (fps={native_fps}, frames={frame_count})")

    n_steps = int(duration_sec * FPS_TARGET)
    frames: List[np.ndarray] = []

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
        raise RuntimeError(f"No frames extracted: {video_path}")

    X = np.stack(frames, axis=0).astype(np.uint8)  # (N,H,W)
    X = X[..., np.newaxis]  # (N,H,W,1)
    return X, duration_sec


def main() -> None:
    if not RAW_VIDEO_DIR.exists():
        raise FileNotFoundError(f"RAW_VIDEO_DIR not found: {RAW_VIDEO_DIR}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")

    # Excel map: session_id -> sheet
    xls = pd.ExcelFile(EXCEL_PATH)
    session_to_sheet: Dict[str, str] = {}
    for sh in xls.sheet_names:
        sid = sheet_to_session_id(sh)
        if sid:
            session_to_sheet[sid] = sh

    if not session_to_sheet:
        raise RuntimeError("No usable sheets found in Excel.")

    # filter sessions (optional)
    targets = sorted(session_to_sheet.keys())
    if SESSIONS_ENV:
        req = [normalize_session_input(x) for x in SESSIONS_ENV.split(",") if x.strip()]
        targets = [t for t in targets if t in set(req)]

    print(f"[INFO] Targets: {len(targets)} sessions")

    manifest_new = []

    for session_id in targets:
        token = normalize_token(session_id)

        # output paths
        y_path = OUT_DIR / f"y_FRAMES_{token}.npy"
        meta_path = OUT_DIR / f"meta_{token}.json"
        x_paths = {v: OUT_DIR / f"X_FRAMES_{v}_{token}.npy" for v in VIEWS}

        if SKIP_EXISTING and y_path.exists() and all(p.exists() for p in x_paths.values()):
            print(f"[SKIP] {session_id} (already exists)")
            continue

        sheet = session_to_sheet[session_id]
        intervals = load_intervals(EXCEL_PATH, sheet)
        if not intervals:
            print(f"[WARN] {session_id}: no intervals in sheet {sheet}, still saving frames with all-0 labels.")

        # find videos
        view_files = find_view_files(RAW_VIDEO_DIR, session_id)
        print(f"\n[DO] {session_id} (sheet={sheet})")
        for v in VIEWS:
            print(f"  - {v}: {view_files[v].name}")

        # extract frames per view
        X_frames: Dict[str, np.ndarray] = {}
        durations = []
        for v in VIEWS:
            Xv, dur = extract_frames_uint8(view_files[v])
            X_frames[v] = Xv
            durations.append(dur)

        # align lengths (trim to min)
        Ns = {v: int(X_frames[v].shape[0]) for v in VIEWS}
        nmin = min(Ns.values())
        if len(set(Ns.values())) != 1:
            print(f"[WARN] Frame count mismatch {Ns} -> trim to {nmin}")
            for v in VIEWS:
                X_frames[v] = X_frames[v][:nmin]

        # build y per-second labels (length nmin)
        y = np.zeros((nmin,), dtype=np.uint8)
        for i in range(nmin):
            t = float(i) / FPS_TARGET
            y[i] = label_at_t(t, intervals)

        # save
        np.save(y_path, y)
        for v in VIEWS:
            np.save(x_paths[v], X_frames[v])

        meta = {
            "session_id": session_id,
            "token": token,
            "sheet": sheet,
            "fps_target": FPS_TARGET,
            "resize_shape": list(RESIZE_SHAPE),
            "views": VIEWS,
            "frames_per_view": {v: int(X_frames[v].shape[0]) for v in VIEWS},
            "pos_frames": int(np.sum(y == 1)),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Saved frames+labels: {token} (N={nmin}, pos_frames={int(np.sum(y==1))})")
        manifest_new.append(meta)

    # append manifest
    if manifest_new:
        if MANIFEST_PATH.exists():
            try:
                old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                if not isinstance(old, list):
                    old = []
            except Exception:
                old = []
        else:
            old = []
        old.extend(manifest_new)
        MANIFEST_PATH.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"\n[OK] Updated manifest: {MANIFEST_PATH}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
