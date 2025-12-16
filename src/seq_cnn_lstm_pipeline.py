"""
Sequence CNN+LSTM pipeline for rodent seizure detection using multi-view videos (TOP, SIDE, SIDE2).

- Scans RAW_VIDEO_DIR for mp4 files.
- Builds session_id -> {TOP, SIDE, SIDE2} map.
- Reads labels from Excel (sheet per session) and converts time ranges to per-frame labels.
- Extracts sequences at 1 fps, resizes to 128x128, converts to grayscale (optional),
  concatenates 3 views horizontally into one frame (H x (3W) x 1).
- Builds tf.data.Dataset of (sequence, label) windows.
- Trains TimeDistributed CNN + LSTM classifier.
- Evaluates and saves model + metrics.

Default paths are aligned with your previous logs:
  RAW_VIDEO_DIR = /home/kimlabmouse/gcs/inputs
  EXCEL_PATH    = /home/kimlabmouse/seizure-detector/data/seizure_stage.xlsx

Author: ChatGPT
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Video
import cv2

# Excel
import pandas as pd

# ML
import tensorflow as tf


# ==============================
# Configuration
# ==============================

@dataclass
class Config:
    # Input
    raw_video_dir: str = "/home/kimlabmouse/gcs/inputs"
    excel_path: str = "/home/kimlabmouse/seizure-detector/data/seizure_stage.xlsx"

    # Views required for a session to be considered valid
    required_views: Tuple[str, ...] = ("TOP", "SIDE", "SIDE2")

    # Video preprocessing
    fps: int = 1
    frame_h: int = 128
    frame_w: int = 128
    grayscale: bool = True  # make single channel for each view
    concat_mode: str = "hstack"  # "hstack" -> (H, 3W, C)

    # Windowing
    seq_len: int = 30            # number of frames per sequence
    stride: int = 5              # sliding window stride in frames
    positive_if_any: bool = True # label window as positive if any frame is positive

    # Label logic
    seizure_positive_stages: Tuple[int, ...] = (1, 2, 3, 4, 5)  # treat stage >=1 as positive by default

    # Output dirs
    processed_dir: str = "data/processed_seq"
    results_dir: str = "results/seq_cnn_lstm"

    # Training
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    val_split: float = 0.2
    seed: int = 42

    # Misc
    overwrite: bool = False
    session_filter: Optional[str] = None  # if set, only process sessions containing this substring


# ==============================
# Utilities
# ==============================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_all_videos(raw_dir: str) -> List[Path]:
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"RAW_VIDEO_DIR does not exist: {raw}")
    videos = sorted(list(raw.rglob("*.mp4")))
    return videos


def normalize_view_name(name: str) -> Optional[str]:
    n = name.lower()
    # Common patterns you have used:
    # - webcamup => TOP
    # - webcamside1 => SIDE
    # - webcamside2 => SIDE2
    if "webcamup" in n or "top" in n:
        return "TOP"
    if "webcamside1" in n or "side1" in n or re.search(r"\bside\b", n):
        # If "side2" is present, it should map to SIDE2, so guard:
        if "side2" in n or "webcamside2" in n:
            return "SIDE2"
        return "SIDE"
    if "webcamside2" in n or "side2" in n:
        return "SIDE2"
    return None


def parse_session_id_from_filename(path: Path) -> Optional[str]:
    """
    Extract session id from filename.

    This function is intentionally conservative and can be adjusted to your naming scheme.
    It tries to detect patterns like:
      - "112625 F1"
      - "061725 POST KA F1"
      - "POST KA061725 F1-....mp4"
      - and variants with suffix like "_B"
    """
    stem = path.stem

    # Example: POST KA061725 F1-webcamside1
    m = re.search(r"(?:KA)?(\d{6})\s*([FM]\d(?:_B)?)", stem)
    if m:
        date6 = m.group(1)
        animal = m.group(2).replace("_", "")
        # You used styles like "061725 POST KA F1" in sheets;
        # but in some logs you had "112625 F1". We'll standardize to "MMDDYY <animal>".
        # date6 is assumed MMDDYY.
        return f"{date6} {animal}"

    # Example: "112625 F1" already in stem
    m2 = re.search(r"(\d{6})\s*([FM]\d(?:_B)?)", stem)
    if m2:
        return f"{m2.group(1)} {m2.group(2).replace('_','')}"

    return None


def build_session_map(video_paths: List[Path]) -> Dict[str, Dict[str, Path]]:
    sessions: Dict[str, Dict[str, Path]] = {}
    for p in video_paths:
        session_id = parse_session_id_from_filename(p)
        if not session_id:
            continue
        view = normalize_view_name(p.name)
        if not view:
            continue
        sessions.setdefault(session_id, {})[view] = p
    return sessions


def filter_and_check_sessions(
    session_map: Dict[str, Dict[str, Path]],
    required_views: Tuple[str, ...],
    session_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for sid in sorted(session_map.keys()):
        if session_filter and session_filter not in sid:
            continue
        views = session_map[sid]
        missing = [v for v in required_views if v not in views]
        if missing:
            continue
        out[sid] = views
    return out


# ==============================
# Excel label parsing
# ==============================

def time_to_seconds(t: str) -> Optional[int]:
    """
    Convert HH:MM:SS or MM:SS or SS to seconds.
    Returns None if parsing fails.
    """
    if t is None:
        return None
    s = str(t).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        parts_i = [int(float(x)) for x in parts]
    except Exception:
        return None
    if len(parts_i) == 3:
        h, m, sec = parts_i
        return h * 3600 + m * 60 + sec
    if len(parts_i) == 2:
        m, sec = parts_i
        return m * 60 + sec
    if len(parts_i) == 1:
        return parts_i[0]
    return None


def read_excel_sheets(excel_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all sheets into a dict.
    Requires openpyxl.
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    sheets: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=name, engine="openpyxl")
        sheets[name.strip()] = df
    return sheets


def match_sheet_for_session(sheet_names: List[str], session_id: str) -> Optional[str]:
    """
    Match session_id to a sheet name.

    session_id is normalized as "MMDDYY <animal>" like "112625 F1" or "112625 F1B".
    We'll try:
      - direct substring match
      - removing spaces
      - handling "_B" variants (F1_B vs F1B)
    """
    sid = session_id.strip()
    sid_compact = re.sub(r"\s+", "", sid).upper()

    candidates = []
    for sh in sheet_names:
        sh_u = sh.upper()
        sh_compact = re.sub(r"\s+", "", sh_u)
        # direct
        if sid.upper() in sh_u:
            candidates.append(sh)
            continue
        # compact
        if sid_compact in sh_compact:
            candidates.append(sh)
            continue

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # pick shortest sheet name as heuristic
        candidates.sort(key=lambda x: len(x))
        return candidates[0]
    return None


def build_frame_labels_from_sheet(
    df: pd.DataFrame,
    total_frames: int,
    fps: int,
    positive_stages: Tuple[int, ...],
) -> np.ndarray:
    """
    Build per-frame binary labels from a sheet.

    Expected columns (flexible):
      - start time: "Start", "start", "Start Time", "From"
      - end time: "End", "end", "End Time", "To"
      - stage: "Stage", "stage", "Seizure Stage", "Label"

    If stage is missing, all intervals are treated as positive.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    def find_col(options: List[str]) -> Optional[str]:
        for opt in options:
            if opt in cols:
                return cols[opt]
        return None

    c_start = find_col(["start", "start time", "from", "begin"])
    c_end = find_col(["end", "end time", "to", "finish"])
    c_stage = find_col(["stage", "seizure stage", "label"])

    if c_start is None or c_end is None:
        # Cannot interpret intervals. Return all zeros.
        return np.zeros((total_frames,), dtype=np.int32)

    y = np.zeros((total_frames,), dtype=np.int32)

    for _, row in df.iterrows():
        s0 = time_to_seconds(row.get(c_start))
        s1 = time_to_seconds(row.get(c_end))
        if s0 is None or s1 is None:
            continue
        if s1 < s0:
            continue

        stage_ok = True
        if c_stage is not None:
            try:
                stage_val = int(float(row.get(c_stage)))
                stage_ok = stage_val in positive_stages
            except Exception:
                # if stage is unparseable, ignore this interval
                stage_ok = False

        if not stage_ok:
            continue

        f0 = int(math.floor(s0 * fps))
        f1 = int(math.ceil(s1 * fps))
        f0 = max(0, min(total_frames - 1, f0))
        f1 = max(0, min(total_frames - 1, f1))
        if f1 < f0:
            continue
        y[f0 : f1 + 1] = 1

    return y


# ==============================
# Video extraction
# ==============================

def get_video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def sample_video_frames(
    path: Path,
    fps_out: int,
    frame_h: int,
    frame_w: int,
    grayscale: bool,
) -> np.ndarray:
    """
    Sample frames from a video at fps_out.
    Returns array shape: (T, H, W, C) where C=1 if grayscale else 3.
    Sampling is done by using video fps and stepping.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in is None or fps_in <= 0:
        fps_in = 30.0  # fallback

    step = max(1, int(round(float(fps_in) / float(fps_out))))

    frames: List[np.ndarray] = []
    idx = 0
    grabbed = True
    while grabbed:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if idx % step == 0:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
                frame = frame[:, :, None]  # H,W,1
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from: {path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    return arr


def concat_views(
    top: np.ndarray,
    side: np.ndarray,
    side2: np.ndarray,
    mode: str = "hstack",
) -> np.ndarray:
    """
    top/side/side2: (T,H,W,C) each, same T.
    Return: (T,H,3W,C) if mode == "hstack"
    """
    T = min(top.shape[0], side.shape[0], side2.shape[0])
    top = top[:T]
    side = side[:T]
    side2 = side2[:T]

    if mode == "hstack":
        return np.concatenate([top, side, side2], axis=2)  # width axis
    raise ValueError(f"Unknown concat_mode: {mode}")


def save_session_arrays(
    out_dir: Path,
    session_id: str,
    X: np.ndarray,
    y_frame: np.ndarray,
    overwrite: bool,
) -> Tuple[Path, Path]:
    ensure_dir(out_dir)
    safe = re.sub(r"[^A-Za-z0-9_ -]", "_", session_id).replace(" ", "_")
    x_path = out_dir / f"X_{safe}.npy"
    y_path = out_dir / f"y_{safe}.npy"

    if (x_path.exists() or y_path.exists()) and not overwrite:
        return x_path, y_path

    np.save(x_path, X)
    np.save(y_path, y_frame)
    return x_path, y_path


# ==============================
# Windowing -> Dataset
# ==============================

def make_windows(
    X: np.ndarray,
    y_frame: np.ndarray,
    seq_len: int,
    stride: int,
    positive_if_any: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (T,H,W,C)
    y_frame: (T,)
    Returns:
      Xw: (N, seq_len, H, W, C)
      yw: (N,)
    """
    T = X.shape[0]
    if T < seq_len:
        return np.zeros((0, seq_len, *X.shape[1:]), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X_list = []
    y_list = []
    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        xw = X[start:end]
        yf = y_frame[start:end]
        yw = 1 if (np.any(yf == 1) if positive_if_any else (np.mean(yf) >= 0.5)) else 0
        X_list.append(xw)
        y_list.append(yw)

    Xw = np.stack(X_list, axis=0).astype(np.float32) / 255.0
    yw = np.array(y_list, dtype=np.int32)
    return Xw, yw


def build_dataset_from_sessions(
    processed_dir: Path,
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load all per-session window arrays from processed_dir and build train/val tf.data datasets.
    """
    x_files = sorted(processed_dir.glob("X_*.npy"))
    y_files = sorted(processed_dir.glob("y_*.npy"))

    y_map = {f.name.replace("y_", "X_"): f for f in y_files}

    X_all = []
    y_all = []

    for xf in x_files:
        yf = y_map.get(xf.name)
        if yf is None:
            continue
        Xw = np.load(xf)  # (N, seq, H, W, C)
        yw = np.load(yf)  # (N,)
        if Xw.shape[0] == 0:
            continue
        X_all.append(Xw)
        y_all.append(yw)

    if len(X_all) == 0:
        raise RuntimeError(f"No window data found in: {processed_dir}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Shuffle + split
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n_val = int(round(X.shape[0] * val_split))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    def make_ds(Xn: np.ndarray, yn: np.ndarray) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((Xn, yn))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return make_ds(X_tr, y_tr), make_ds(X_val, y_val)


# ==============================
# Model (TimeDistributed CNN + LSTM)
# ==============================

def build_seq_cnn_lstm_model(
    seq_len: int,
    H: int,
    W: int,
    C: int,
    lr: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, H, W, C), name="video_seq")

    # Per-frame CNN
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")
    )(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2)))(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
    )(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2)))(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
    )(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2)))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Temporal modeling
    x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="seizure_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="seq_cnn_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ==============================
# End-to-end pipeline
# ==============================

def preprocess_all_sessions(cfg: Config) -> Tuple[int, int]:
    """
    Preprocess sessions:
      - extract frames from each view
      - read label sheet
      - convert to per-frame labels aligned to sampled fps
      - build windows and save per-session window arrays to processed_dir

    Returns: (num_sessions_processed, num_windows_total)
    """
    set_seed(cfg.seed)

    videos = find_all_videos(cfg.raw_video_dir)
    session_map = build_session_map(videos)
    ok_sessions = filter_and_check_sessions(session_map, cfg.required_views, cfg.session_filter)

    if len(ok_sessions) == 0:
        raise RuntimeError("No valid sessions found with required views (TOP, SIDE, SIDE2).")

    # Load all sheets once
    sheets = read_excel_sheets(cfg.excel_path)
    sheet_names = list(sheets.keys())

    processed_dir = Path(cfg.processed_dir)
    ensure_dir(processed_dir)

    sessions_done = 0
    windows_total = 0

    for sid in sorted(ok_sessions.keys()):
        views = ok_sessions[sid]

        sh = match_sheet_for_session(sheet_names, sid)
        if sh is None:
            # No labels found; skip
            continue

        # Extract view frames
        top = sample_video_frames(views["TOP"], cfg.fps, cfg.frame_h, cfg.frame_w, cfg.grayscale)
        side = sample_video_frames(views["SIDE"], cfg.fps, cfg.frame_h, cfg.frame_w, cfg.grayscale)
        side2 = sample_video_frames(views["SIDE2"], cfg.fps, cfg.frame_h, cfg.frame_w, cfg.grayscale)

        X = concat_views(top, side, side2, mode=cfg.concat_mode)  # (T,H,3W,C)

        # Build per-frame labels aligned to sampled fps (1 fps => 1 frame per sec)
        df = sheets[sh]
        y_frame = build_frame_labels_from_sheet(
            df=df,
            total_frames=X.shape[0],
            fps=cfg.fps,
            positive_stages=cfg.seizure_positive_stages,
        )

        # Windowing
        Xw, yw = make_windows(X, y_frame, cfg.seq_len, cfg.stride, cfg.positive_if_any)

        # Save per-session windows
        out_session_dir = processed_dir / "windows"
        ensure_dir(out_session_dir)
        safe = re.sub(r"[^A-Za-z0-9_ -]", "_", sid).replace(" ", "_")
        x_path = out_session_dir / f"X_{safe}.npy"
        y_path = out_session_dir / f"y_{safe}.npy"

        if (x_path.exists() or y_path.exists()) and not cfg.overwrite:
            # Count existing windows for bookkeeping
            try:
                existing = np.load(x_path, mmap_mode="r")
                windows_total += int(existing.shape[0])
            except Exception:
                pass
            continue

        np.save(x_path, Xw)
        np.save(y_path, yw)

        sessions_done += 1
        windows_total += int(Xw.shape[0])

    return sessions_done, windows_total


def train_and_evaluate(cfg: Config) -> Dict[str, float]:
    set_seed(cfg.seed)

    processed_windows_dir = Path(cfg.processed_dir) / "windows"
    if not processed_windows_dir.exists():
        raise FileNotFoundError(
            f"Processed windows directory not found: {processed_windows_dir}. "
            f"Run preprocess first."
        )

    train_ds, val_ds = build_dataset_from_sessions(
        processed_dir=processed_windows_dir,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    # infer input shape from one batch
    for xb, yb in train_ds.take(1):
        # xb: (B, seq, H, W, C)
        _, seq_len, H, W, C = xb.shape
        seq_len = int(seq_len)
        H = int(H); W = int(W); C = int(C)

    model = build_seq_cnn_lstm_model(seq_len=seq_len, H=H, W=W, C=C, lr=cfg.lr)

    results_dir = Path(cfg.results_dir)
    ensure_dir(results_dir)

    # Save config used
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(results_dir / "train_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    eval_metrics = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    metrics_dict = {k: float(v) for k, v in zip(metric_names, eval_metrics)}

    # Save final model too
    model.save(str(results_dir / "final_model.keras"))

    # Save metrics
    with open(results_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save history summary
    hist_path = results_dir / "history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    return metrics_dict


# ==============================
# CLI
# ==============================

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequence CNN+LSTM multi-view seizure detection pipeline")

    p.add_argument("--mode", type=str, default="all", choices=["preprocess", "train", "all"])
    p.add_argument("--raw_video_dir", type=str, default=Config.raw_video_dir)
    p.add_argument("--excel_path", type=str, default=Config.excel_path)

    p.add_argument("--processed_dir", type=str, default=Config.processed_dir)
    p.add_argument("--results_dir", type=str, default=Config.results_dir)

    p.add_argument("--fps", type=int, default=Config.fps)
    p.add_argument("--frame_h", type=int, default=Config.frame_h)
    p.add_argument("--frame_w", type=int, default=Config.frame_w)
    p.add_argument("--grayscale", action="store_true", default=Config.grayscale)
    p.add_argument("--no_grayscale", action="store_false", dest="grayscale")

    p.add_argument("--seq_len", type=int, default=Config.seq_len)
    p.add_argument("--stride", type=int, default=Config.stride)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--val_split", type=float, default=Config.val_split)

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--overwrite", action="store_true", default=Config.overwrite)
    p.add_argument("--session_filter", type=str, default=None)

    return p


def main() -> None:
    args = make_argparser().parse_args()

    cfg = Config(
        raw_video_dir=args.raw_video_dir,
        excel_path=args.excel_path,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        fps=args.fps,
        frame_h=args.frame_h,
        frame_w=args.frame_w,
        grayscale=args.grayscale,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        overwrite=args.overwrite,
        session_filter=args.session_filter,
    )

    # Basic existence checks to avoid long crashes
    if not Path(cfg.raw_video_dir).exists():
        raise FileNotFoundError(f"RAW_VIDEO_DIR not found: {cfg.raw_video_dir}")
    if not Path(cfg.excel_path).exists():
        raise FileNotFoundError(f"EXCEL_PATH not found: {cfg.excel_path}")

    ensure_dir(cfg.processed_dir)
    ensure_dir(cfg.results_dir)

    if args.mode in ("preprocess", "all"):
        sessions_done, windows_total = preprocess_all_sessions(cfg)
        print(f"[OK] Preprocess done. Sessions processed: {sessions_done}, total windows: {windows_total}")

    if args.mode in ("train", "all"):
        metrics = train_and_evaluate(cfg)
        print("[OK] Train/Eval done. Metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}")


if __name__ == "__main__":
    main()
