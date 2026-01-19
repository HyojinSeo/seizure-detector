#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def hms_to_sec(t: str) -> float:
    """
    Convert HH:MM:SS or MM:SS or H:MM:SS to seconds
    """
    parts = str(t).strip().split(":")
    if len(parts) == 2:          # MM:SS
        m, s = parts
        return int(m) * 60 + int(s)
    elif len(parts) == 3:        # H:MM:SS or HH:MM:SS
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Invalid time format: {t}")


def sec_to_hhmmss(sec: float) -> str:
    sec_int = int(round(sec))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_label_intervals(label_xlsx: str) -> List[Tuple[float, float]]:
    """
    Label Excel format:
      Time | Description | Stage
      0:04 - 0:08
      1:00:14 - 1:00:22
    """
    df = pd.read_excel(label_xlsx)
    intervals = []
    for _, row in df.iterrows():
        if "Time" not in df.columns:
            raise ValueError("Label Excel must have a 'Time' column.")
        if not isinstance(row["Time"], str):
            continue
        if "-" not in row["Time"]:
            continue
        left, right = row["Time"].split("-", 1)
        start = hms_to_sec(left.strip())
        end = hms_to_sec(right.strip())
        if end < start:
            start, end = end, start
        intervals.append((start, end))
    return intervals


def load_pred_intervals(pred_xlsx: str) -> List[Tuple[float, float]]:
    """
    Prediction Excel format:
      start_hms | end_hms | mean_prob
      00:20:24  | 00:20:40
    """
    df = pd.read_excel(pred_xlsx)

    # Prefer start_sec/end_sec if present, else use start_hms/end_hms
    if "start_sec" in df.columns and "end_sec" in df.columns:
        intervals = list(zip(df["start_sec"].astype(float), df["end_sec"].astype(float)))
        return intervals

    if "start_hms" not in df.columns or "end_hms" not in df.columns:
        raise ValueError("Prediction Excel must have start_sec/end_sec or start_hms/end_hms columns.")

    intervals = []
    for _, row in df.iterrows():
        start = hms_to_sec(row["start_hms"])
        end = hms_to_sec(row["end_hms"])
        if end < start:
            start, end = end, start
        intervals.append((start, end))
    return intervals


def plot_intervals(ax, intervals: List[Tuple[float, float]], y: float, height: float, label: str):
    # broken_barh: list of (xmin, width)
    bars = [(s, max(0.0, e - s)) for s, e in intervals]
    ax.broken_barh(bars, (y, height), label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_xlsx", required=True, type=str, help="Label Excel (e.g., 010626F1.xlsx)")
    ap.add_argument("--pred_xlsx", required=True, type=str, help="Prediction Excel (e.g., KA010626_F1_intervals.xlsx)")
    ap.add_argument("--out_png", default=None, type=str, help="Optional output PNG path")
    args = ap.parse_args()

    label_intervals = load_label_intervals(args.label_xlsx)
    pred_intervals = load_pred_intervals(args.pred_xlsx)

    if not label_intervals and not pred_intervals:
        raise RuntimeError("No intervals found in both label and prediction files.")

    max_end = 0.0
    if label_intervals:
        max_end = max(max_end, max(e for _, e in label_intervals))
    if pred_intervals:
        max_end = max(max_end, max(e for _, e in pred_intervals))

    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Two rows: label on top, prediction below
    plot_intervals(ax, label_intervals, y=20, height=8, label="Label")
    plot_intervals(ax, pred_intervals, y=5, height=8, label="Prediction")

    ax.set_ylim(0, 35)
    ax.set_xlim(0, max_end if max_end > 0 else 1)
    ax.set_yticks([9, 24])
    ax.set_yticklabels(["Prediction", "Label"])

    # Make readable x ticks (HH:MM:SS)
    # Pick ~10 ticks
    n_ticks = 10
    step = max(1, int(round(max_end / n_ticks))) if max_end > 0 else 1
    ticks = list(range(0, int(max_end) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sec_to_hhmmss(t) for t in ticks], rotation=0)

    ax.set_xlabel("Time (HH:MM:SS)")
    ax.set_title("Seizure intervals timeline: Label vs Prediction")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linewidth=0.5)

    fig.tight_layout()

    if args.out_png:
        fig.savefig(args.out_png, dpi=200)
        print(f"[OK] Saved plot: {args.out_png}")

    plt.show()


if __name__ == "__main__":
    main()
