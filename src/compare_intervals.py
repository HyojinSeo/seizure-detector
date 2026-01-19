#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


# =========================
# Time utilities
# =========================
def hms_to_sec(t: str) -> float:
    """
    Convert HH:MM:SS or MM:SS or H:MM:SS to seconds
    """
    parts = t.strip().split(":")
    if len(parts) == 2:          # MM:SS
        m, s = parts
        return int(m) * 60 + int(s)
    elif len(parts) == 3:        # H:MM:SS or HH:MM:SS
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        raise ValueError(f"Invalid time format: {t}")


# =========================
# Load label intervals
# =========================
def load_label_intervals(xlsx_path: str):
    """
    Label Excel format:
      Time | Description | Stage
      0:04 - 0:08
      1:00:14 - 1:00:22
    """
    df = pd.read_excel(xlsx_path)

    intervals = []
    for _, row in df.iterrows():
        if not isinstance(row["Time"], str):
            continue

        left, right = row["Time"].split("-")
        start_sec = hms_to_sec(left.strip())
        end_sec = hms_to_sec(right.strip())
        intervals.append((start_sec, end_sec))

    return intervals


# =========================
# Load prediction intervals
# =========================
def load_pred_intervals(xlsx_path: str):
    """
    Prediction Excel format:
      start_hms | end_hms | mean_prob
      00:20:24  | 00:20:40
    """
    df = pd.read_excel(xlsx_path)

    intervals = []
    for _, row in df.iterrows():
        start_sec = hms_to_sec(row["start_hms"])
        end_sec = hms_to_sec(row["end_hms"])
        intervals.append((start_sec, end_sec))

    return intervals


# =========================
# Interval comparison
# =========================
def overlaps(a, b) -> bool:
    """
    a, b: (start_sec, end_sec)
    """
    return not (a[1] <= b[0] or b[1] <= a[0])


def compare_intervals(label_intervals, pred_intervals):
    """
    Interval-level evaluation
    TP: predicted interval overlaps any label interval
    FP: predicted interval overlaps none
    FN: label interval not overlapped by any prediction
    """
    TP = FP = FN = 0
    matched_labels = set()

    for p in pred_intervals:
        hit = False
        for i, l in enumerate(label_intervals):
            if overlaps(p, l):
                hit = True
                matched_labels.add(i)
        if hit:
            TP += 1
        else:
            FP += 1

    for i in range(len(label_intervals)):
        if i not in matched_labels:
            FN += 1

    return TP, FP, FN


# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_xlsx", required=True, type=str, help="Label Excel (.xlsx) path")
    ap.add_argument("--pred_xlsx", required=True, type=str, help="Prediction Excel (.xlsx) path")
    args = ap.parse_args()

    label_intervals = load_label_intervals(args.label_xlsx)
    pred_intervals = load_pred_intervals(args.pred_xlsx)

    TP, FP, FN = compare_intervals(label_intervals, pred_intervals)

    print("===================================")
    print("Interval-level evaluation")
    print("===================================")
    print(f"Label intervals : {len(label_intervals)}")
    print(f"Pred intervals  : {len(pred_intervals)}")
    print("-----------------------------------")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0

    print("-----------------------------------")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
