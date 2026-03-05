#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def hms_to_sec(t: str) -> float:
    parts = str(t).strip().split(":")
    if len(parts) == 2:  # MM:SS
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:  # HH:MM:SS (or H:MM:SS)
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Invalid time format: {t}")


def sec_to_hhmmss(sec: float) -> str:
    sec_int = int(round(float(sec)))
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_gt_intervals_from_sheet(gt_xlsx: str, sheet: str) -> List[Tuple[float, float]]:
    """
    GT Excel (multi-sheet) format:
      column 'Time' containing strings like "0:04 - 0:08" or "1:00:14 - 1:00:22"
    """
    df = pd.read_excel(gt_xlsx, sheet_name=sheet)

    if "Time" not in df.columns:
        raise ValueError(f"GT sheet '{sheet}' must have a 'Time' column.")

    intervals: List[Tuple[float, float]] = []
    for v in df["Time"].dropna().tolist():
        if not isinstance(v, str):
            continue
        if "-" not in v:
            continue
        left, right = v.split("-", 1)
        start = hms_to_sec(left.strip())
        end = hms_to_sec(right.strip())
        if end < start:
            start, end = end, start
        intervals.append((float(start), float(end)))

    return intervals


def load_pred_intervals(pred_xlsx: str) -> List[Tuple[float, float]]:
    """
    Prediction Excel format (from infer_timeline_latefusion*.py):
      start_sec | end_sec | mean_prob | (optional) start_hms/end_hms
    """
    df = pd.read_excel(pred_xlsx)
    if df.empty:
        return []

    if "start_sec" in df.columns and "end_sec" in df.columns:
        return list(zip(df["start_sec"].astype(float), df["end_sec"].astype(float)))

    # fallback
    if "start_hms" in df.columns and "end_hms" in df.columns:
        out = []
        for _, row in df.iterrows():
            s = hms_to_sec(row["start_hms"])
            e = hms_to_sec(row["end_hms"])
            if e < s:
                s, e = e, s
            out.append((float(s), float(e)))
        return out

    raise ValueError("Prediction Excel must have start_sec/end_sec or start_hms/end_hms columns.")


def plot_intervals(ax, intervals: List[Tuple[float, float]], y: float, height: float, label: str):
    bars = [(s, max(0.0, e - s)) for s, e in intervals]
    if bars:
        ax.broken_barh(bars, (y, height), label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_xlsx", required=True, type=str, help="GT Excel (multi-sheet), e.g. data/seizure_stage.xlsx")
    ap.add_argument("--sheet", required=True, type=str, help="Sheet name, e.g. 010626F1_B")
    ap.add_argument("--pred_xlsx", required=True, type=str, help="Prediction intervals xlsx")
    ap.add_argument("--out_png", required=True, type=str, help="Output PNG path")
    ap.add_argument("--title", default=None, type=str, help="Optional plot title")
    args = ap.parse_args()

    gt_intervals = load_gt_intervals_from_sheet(args.gt_xlsx, args.sheet)
    pred_intervals = load_pred_intervals(args.pred_xlsx)

    if not gt_intervals and not pred_intervals:
        raise RuntimeError("No intervals found in both GT and prediction files.")

    max_end = 0.0
    if gt_intervals:
        max_end = max(max_end, max(e for _, e in gt_intervals))
    if pred_intervals:
        max_end = max(max_end, max(e for _, e in pred_intervals))

    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Label on top, Prediction below
    plot_intervals(ax, gt_intervals, y=20, height=8, label="GT (Label)")
    plot_intervals(ax, pred_intervals, y=5, height=8, label="Prediction")

    ax.set_ylim(0, 35)
    ax.set_xlim(0, max_end if max_end > 0 else 1)
    ax.set_yticks([9, 24])
    ax.set_yticklabels(["Prediction", "GT"])

    # x ticks
    n_ticks = 10
    step = max(1, int(round(max_end / n_ticks))) if max_end > 0 else 1
    ticks = list(range(0, int(max_end) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([sec_to_hhmmss(t) for t in ticks], rotation=0)

    ax.set_xlabel("Time (HH:MM:SS)")
    ax.set_title(args.title if args.title else f"Timeline: {args.sheet} (GT vs Pred)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linewidth=0.5)
    fig.tight_layout()

    out_png = args.out_png
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"[OK] Saved plot: {out_png}")

    # optional: show if you want interactive
    # plt.show()


if __name__ == "__main__":
    from pathlib import Path
    main()
