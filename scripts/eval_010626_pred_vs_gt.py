#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# -------------------------
# Time parsing
# -------------------------
def hms_to_sec(t: str) -> Optional[float]:
    if t is None:
        return None
    s = str(t).strip()
    if not s or ":" not in s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:  # MM:SS
            m, sec = parts
            return int(m) * 60 + float(sec)
        if len(parts) == 3:  # H:MM:SS
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None
    return None


def parse_time_range_cell(cell: str) -> Optional[Tuple[float, float]]:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or "-" not in s:
        return None
    left, right = s.split("-", 1)
    a = hms_to_sec(left.strip())
    b = hms_to_sec(right.strip())
    if a is None or b is None:
        return None
    if b < a:
        a, b = b, a
    return float(a), float(b)


def merge_intervals(intervals: List[Tuple[float, float]], gap_tol: float = 0.0) -> List[Tuple[float, float]]:
    """Sort + merge overlapping/close intervals (gap_tol seconds)."""
    if not intervals:
        return []
    arr = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [arr[0]]
    for s, e in arr[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap_tol:  # overlap or close enough
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def load_gt_intervals(gt_xlsx: Path, sheet: str) -> List[Tuple[float, float]]:
    df = pd.read_excel(gt_xlsx, sheet_name=sheet)
    if "Time" not in df.columns:
        raise RuntimeError(f"GT sheet '{sheet}' missing 'Time' column")

    out: List[Tuple[float, float]] = []
    for cell in df["Time"].tolist():
        rng = parse_time_range_cell(cell)
        if rng is not None:
            out.append(rng)
    return merge_intervals(out, gap_tol=0.0)


def load_pred_intervals(pred_xlsx: Path) -> List[Tuple[float, float]]:
    if not pred_xlsx.exists():
        return []
    df = pd.read_excel(pred_xlsx)
    if df.empty:
        return []

    out: List[Tuple[float, float]] = []
    if "start_sec" in df.columns and "end_sec" in df.columns:
        for _, r in df.iterrows():
            try:
                s = float(r["start_sec"]); e = float(r["end_sec"])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append((s, e))
        return merge_intervals(out, gap_tol=0.0)

    if "start_hms" in df.columns and "end_hms" in df.columns:
        for _, r in df.iterrows():
            s = hms_to_sec(r["start_hms"])
            e = hms_to_sec(r["end_hms"])
            if s is None or e is None:
                continue
            if e < s:
                s, e = e, s
            out.append((float(s), float(e)))
        return merge_intervals(out, gap_tol=0.0)

    return []


# -------------------------
# Metrics
# -------------------------
def overlap_len(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def interval_level_metrics(
    gt: List[Tuple[float, float]],
    pred: List[Tuple[float, float]],
    min_overlap_sec: float = 1.0,
) -> Dict[str, float]:
    """
    Interval-level:
      - GT hit: a GT interval is hit if any pred overlaps >= min_overlap_sec
      - Pred hit: a pred interval is hit if any GT overlaps >= min_overlap_sec
    """
    gt_hit = sum(any(overlap_len(g, p) >= min_overlap_sec for p in pred) for g in gt)
    pred_hit = sum(any(overlap_len(g, p) >= min_overlap_sec for g in gt) for p in pred)

    recall = gt_hit / len(gt) if gt else 0.0
    precision = pred_hit / len(pred) if pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "gt_n": float(len(gt)),
        "pred_n": float(len(pred)),
        "gt_hit": float(gt_hit),
        "pred_hit": float(pred_hit),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def second_level_metrics(
    gt: List[Tuple[float, float]],
    pred: List[Tuple[float, float]],
    max_sec: Optional[int] = None,
    step_sec: int = 1,
) -> Dict[str, float]:
    """
    Second-level (timepoint):
      Build binary label for each timepoint t = 0,step,2*step,...,max
      t is positive if t is within any interval [start,end).
    """
    if max_sec is None:
        m = 0.0
        if gt:
            m = max(m, max(e for _, e in gt))
        if pred:
            m = max(m, max(e for _, e in pred))
        max_sec = int(np.ceil(m))

    if max_sec <= 0:
        return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    times = np.arange(0, max_sec + 1, step_sec, dtype=np.int32)

    def mark(intervals: List[Tuple[float, float]]) -> np.ndarray:
        y = np.zeros_like(times, dtype=np.int8)
        for s, e in intervals:
            # mark t where s <= t < e
            mask = (times >= int(np.floor(s))) & (times < int(np.ceil(e)))
            y[mask] = 1
        return y

    y_gt = mark(gt)
    y_pr = mark(pred)

    tp = int(np.sum((y_gt == 1) & (y_pr == 1)))
    fp = int(np.sum((y_gt == 0) & (y_pr == 1)))
    fn = int(np.sum((y_gt == 1) & (y_pr == 0)))
    tn = int(np.sum((y_gt == 0) & (y_pr == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "max_sec": float(max_sec),
        "step_sec": float(step_sec),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_xlsx", required=True, type=str, help="data/seizure_stage.xlsx")
    ap.add_argument("--pred_dir", required=True, type=str, help="results/infer/010626/all3 (contains *_pred.xlsx)")
    ap.add_argument("--min_overlap_sec", type=float, default=1.0)
    ap.add_argument("--step_sec", type=int, default=1)
    args = ap.parse_args()

    gt_xlsx = Path(args.gt_xlsx).expanduser()
    pred_dir = Path(args.pred_dir).expanduser()

    sessions = [
        ("010626F1",   pred_dir / "010626_F1_pred.xlsx"),
        ("010626F1_B", pred_dir / "010626_F1_B_pred.xlsx"),
        ("010626F2",   pred_dir / "010626_F2_pred.xlsx"),
        ("010626F2_B", pred_dir / "010626_F2_B_pred.xlsx"),
        ("010626M1",   pred_dir / "010626_M1_pred.xlsx"),
        ("010626M1_B", pred_dir / "010626_M1_B_pred.xlsx"),
        ("010626M2",   pred_dir / "010626_M2_pred.xlsx"),
        ("010626M2_B", pred_dir / "010626_M2_B_pred.xlsx"),
    ]

    rows = []
    for sheet, pred_path in sessions:
        gt = load_gt_intervals(gt_xlsx, sheet)
        pred = load_pred_intervals(pred_path)

        im = interval_level_metrics(gt, pred, min_overlap_sec=args.min_overlap_sec)
        sm = second_level_metrics(gt, pred, step_sec=args.step_sec)

        rows.append({
            "sheet": sheet,
            "pred_file": pred_path.name,
            "GT_intervals": int(im["gt_n"]),
            "Pred_intervals": int(im["pred_n"]),
            "Interval_P": im["precision"],
            "Interval_R": im["recall"],
            "Interval_F1": im["f1"],
            "Second_P": sm["precision"],
            "Second_R": sm["recall"],
            "Second_F1": sm["f1"],
            "TP": int(sm["tp"]),
            "FP": int(sm["fp"]),
            "FN": int(sm["fn"]),
        })

    df = pd.DataFrame(rows)

    print("\n==================== 010626 Pred vs GT ====================")
    print(f"GT: {gt_xlsx}")
    print(f"Pred dir: {pred_dir}")
    print(f"Interval-level: min_overlap_sec={args.min_overlap_sec}")
    print(f"Second-level: step_sec={args.step_sec}")
    print("-----------------------------------------------------------")
    print(df[[
        "sheet",
        "GT_intervals", "Pred_intervals",
        "Interval_P", "Interval_R", "Interval_F1",
        "Second_P", "Second_R", "Second_F1",
        "TP", "FP", "FN",
    ]].round(4).to_string(index=False))

    print("\nMacro average (mean over sessions):")
    print(df[["Interval_P","Interval_R","Interval_F1","Second_P","Second_R","Second_F1"]].mean().round(4).to_string())

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
