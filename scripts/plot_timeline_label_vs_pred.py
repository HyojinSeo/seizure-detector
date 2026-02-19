#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def hms_to_sec(t: str) -> float:
    """
    Convert HH:MM:SS or MM:SS or H:MM:SS to seconds.
    Accepts strings like:
      "0:04", "00:04", "1:00:14", "01:00:14"
    """
    parts = str(t).strip()
    if not parts:
        raise ValueError("Empty time string")

    parts = parts.split(":")
    if len(parts) == 2:  # MM:SS
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:  # H:MM:SS
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ValueError(f"Invalid time format: {t}")


def sec_to_hhmmss(sec: float) -> str:
    sec_int = int(round(sec))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_range_cell(cell: str) -> Optional[Tuple[float, float]]:
    """
    Parse "start - end" from the Time column.
    Example: "0:04 - 0:08" or "1:00:14 - 1:00:22"
    Returns (start_sec, end_sec) or None if not parseable.
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or "-" not in s:
        return None

    left, right = s.split("-", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None

    try:
        start = hms_to_sec(left)
        end = hms_to_sec(right)
    except Exception:
        return None

    if end < start:
        start, end = end, start
    return float(start), float(end)


def load_label_intervals(label_xlsx: str, sheet_name: str) -> List[Tuple[float, float]]:
    """
    Label Excel format (sheet):
      Time | Description | Stage | ...
      "0:04 - 0:08"
      "1:00:14 - 1:00:22"
    """
    df = pd.read_excel(label_xlsx, sheet_name=sheet_name)

    if "Time" not in df.columns:
        raise ValueError(f"Label sheet '{sheet_name}' must have a 'Time' column.")

    intervals: List[Tuple[float, float]] = []
    for cell in df["Time"].tolist():
        rng = parse_time_range_cell(cell)
        if rng is not None:
            intervals.append(rng)

    return intervals


def load_pred_intervals(pred_xlsx: str) -> List[Tuple[float, float]]:
    """
    Prediction Excel format from infer_timeline_latefusion.py:
      start_sec | end_sec | mean_prob | start_hms | end_hms (optional)
    We prefer start_sec/end_sec if present.
    """
    df = pd.read_excel(pred_xlsx)
    if df.empty:
        return []

    if "start_sec" in df.columns and "end_sec" in df.columns:
        out = []
        for _, r in df.iterrows():
            try:
                s = float(r["start_sec"])
                e = float(r["end_sec"])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append((s, e))
        return out

    # fallback to hms columns
    if "start_hms" in df.columns and "end_hms" in df.columns:
        out = []
        for _, r in df.iterrows():
            try:
                s = hms_to_sec(r["start_hms"])
                e = hms_to_sec(r["end_hms"])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append((float(s), float(e)))
        return out

    raise ValueError("Prediction Excel must have start_sec/end_sec or start_hms/end_hms columns.")


def plot_intervals(ax, intervals: List[Tuple[float, float]], y: float, height: float, label: str):
    bars = [(s, max(0.0, e - s)) for s, e in intervals]
    ax.broken_barh(bars, (y, height), label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_xlsx", required=True, type=str, help="Label Excel file (e.g., data/seizure_stage.xlsx)")
    ap.add_argument("--label_sheet", required=True, type=str, help="Sheet name inside label_xlsx (e.g., 010626F1)")
    ap.add_argument("--pred_xlsx", required=True, type=str, help="Prediction Excel from infer_timeline_latefusion.py")
    ap.add_argument("--out_png", default=None, type=str, help="Optional output PNG path")
    ap.add_argument("--no_show", action="store_true", help="If set, do not open an interactive window (useful on servers).")
    ap.add_argument("--title", default=None, type=str, help="Optional plot title override")
    args = ap.parse_args()

    label_intervals = load_label_intervals(args.label_xlsx, args.label_sheet)
    pred_intervals = load_pred_intervals(args.pred_xlsx)

    if not label_intervals and not pred_intervals:
        raise RuntimeError("No intervals found in both label and prediction files.")

    max_end = 0.0
    if label_intervals:
        max_end = max(max_end, max(e for _, e in label_intervals))
    if pred_intervals:
        max_end = max(max_end, max(e for _, e in pred_intervals))

    fig, ax = plt.subplots(figsize=(14, 3.8))

    # Two rows: label on top, prediction below
    plot_intervals(ax, label_intervals, y=20, height=8, label="GT Label")
    plot_intervals(ax, pred_intervals, y=5, height=8, label="Prediction")

    ax.set_ylim(0, 35)
    ax.set_xlim(0, max_end if max_end > 0 else 1)
    ax.set_yticks([9, 24])
    ax.set_yticklabels(["Prediction", "GT Label"])

    # readable x ticks (HH:MM:SS)
    n_ticks = 10
    step = max(1, int(round(max_end / n_ticks))) if max_end > 0 else 1
    ticks = list(range(0, int(max_end) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sec_to_hhmmss(t) for t in ticks], rotation=0)

    ax.set_xlabel("Time (HH:MM:SS)")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"Seizure timeline: {args.label_sheet} (GT vs Prediction)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linewidth=0.5)

    fig.tight_layout()

    if args.out_png:
        out_png = args.out_png
        fig.savefig(out_png, dpi=200)
        print(f"[OK] Saved plot: {out_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
