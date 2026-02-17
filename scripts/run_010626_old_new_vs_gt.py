#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def hms_to_sec(s: str) -> float:
    s = str(s).strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m = int(parts[0]); sec = float(parts[1]); return 60*m + sec
        if len(parts) == 3:
            h = int(parts[0]); m = int(parts[1]); sec = float(parts[2]); return 3600*h + 60*m + sec
    except Exception:
        return None
    return None


def load_gt_intervals(gt_xlsx: Path, sheet: str) -> List[Tuple[float, float]]:
    df = pd.read_excel(gt_xlsx, sheet_name=sheet)
    if "Time" not in df.columns:
        raise RuntimeError(f"GT sheet '{sheet}' missing 'Time' column")
    out = []
    for t in df["Time"].dropna().tolist():
        t = str(t)
        if "-" not in t:
            continue
        a, b = t.split("-", 1)
        s = hms_to_sec(a.strip()); e = hms_to_sec(b.strip())
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        out.append((float(s), float(e)))
    return out


def load_pred_intervals(pred_xlsx: Path) -> List[Tuple[float, float]]:
    if not pred_xlsx.exists():
        return []
    df = pd.read_excel(pred_xlsx)
    if df.empty:
        return []
    if "start_sec" not in df.columns or "end_sec" not in df.columns:
        return []
    return [(float(r["start_sec"]), float(r["end_sec"])) for _, r in df.iterrows()]


def overlap_len(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_metrics(gt, pred, min_overlap_sec: float):
    gt_hit = sum(any(overlap_len(g, p) >= min_overlap_sec for p in pred) for g in gt)
    pred_hit = sum(any(overlap_len(g, p) >= min_overlap_sec for g in gt) for p in pred)
    recall = gt_hit / len(gt) if gt else 0.0
    precision = pred_hit / len(pred) if pred else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    return precision, recall, f1, gt_hit, pred_hit


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    repo = Path.home() / "seizure-detector"
    gt_xlsx = repo / "data" / "seizure_stage.xlsx"
    infer_py = repo / "src" / "infer_timeline_latefusion.py"

    old_model = repo / "results" / "latefusion" / "all3" / "best_model.keras"
    new_model = repo / "results" / "latefusion" / "all3_v2_after_add" / "best_model.keras"

    THRESH = 0.50
    BATCH = 16
    MIN_OVERLAP_SEC = 1.0

    out_old = repo / "results" / "infer" / "010626" / "old_all3"
    out_new = repo / "results" / "infer" / "010626" / "new_all3_v2_after_add"
    out_old.mkdir(parents=True, exist_ok=True)
    out_new.mkdir(parents=True, exist_ok=True)

    sessions = [
        ("KA010626 F1",   "010626F1",   "010626_F1_pred.xlsx"),
        ("KA010626 F1 B", "010626F1_B", "010626_F1_B_pred.xlsx"),
        ("KA010626 F2",   "010626F2",   "010626_F2_pred.xlsx"),
        ("KA010626 F2 B", "010626F2_B", "010626_F2_B_pred.xlsx"),
    ]

    # OLD
    for sess, _, fname in sessions:
        run([
            sys.executable, str(infer_py),
            "--session", sess,
            "--model_path", str(old_model),
            "--threshold", str(THRESH),
            "--batch_size", str(BATCH),
            "--out_xlsx", str(out_old / fname),
        ])

    # NEW
    for sess, _, fname in sessions:
        run([
            sys.executable, str(infer_py),
            "--session", sess,
            "--model_path", str(new_model),
            "--threshold", str(THRESH),
            "--batch_size", str(BATCH),
            "--out_xlsx", str(out_new / fname),
        ])

    # Compare
    rows = []
    for _, sheet, fname in sessions:
        gt = load_gt_intervals(gt_xlsx, sheet)
        pred_old = load_pred_intervals(out_old / fname)
        pred_new = load_pred_intervals(out_new / fname)

        op, orc, of1, ogt_hit, _ = interval_metrics(gt, pred_old, MIN_OVERLAP_SEC)
        np_, nr, nf1, ngt_hit, _ = interval_metrics(gt, pred_new, MIN_OVERLAP_SEC)

        rows.append({
            "sheet": sheet,
            "GT_n": len(gt),
            "old_pred_n": len(pred_old),
            "old_prec": op, "old_rec": orc, "old_f1": of1, "old_GT_hit": ogt_hit,
            "new_pred_n": len(pred_new),
            "new_prec": np_, "new_rec": nr, "new_f1": nf1, "new_GT_hit": ngt_hit,
            "delta_rec": nr - orc,
            "delta_f1": nf1 - of1,
        })

    df = pd.DataFrame(rows)
    print("\n==================== 010626: OLD vs NEW vs GT (interval-level) ====================")
    print(f"threshold={THRESH:.2f}, min_overlap_sec={MIN_OVERLAP_SEC:.1f}, batch_size={BATCH}")
    print(df.round(4).to_string(index=False))

    print("\nMacro avg:")
    print(df[["old_prec","new_prec","old_rec","new_rec","delta_rec","old_f1","new_f1","delta_f1"]].mean().round(4))

    print("\nSaved predictions:")
    print(f"  OLD -> {out_old}")
    print(f"  NEW -> {out_new}")


if __name__ == "__main__":
    main()
