#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path.home() / "seizure-detector"
GT = REPO / "data" / "seizure_stage.xlsx"

MODELS = [
    "all3_dyn_16_4",
    "all3_dyn_4_2",
    "all3_dyn_8_4",
]

def run_eval_to_df(pred_dir: Path):
    """
    Run eval script and parse its printed table into pandas DataFrame.
    """
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "eval_010626_pred_vs_gt.py"),
        "--gt_xlsx", str(GT),
        "--pred_dir", str(pred_dir),
        "--min_overlap_sec", "1.0",
        "--step_sec", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = result.stdout

    # Extract lines containing session rows
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("010626"):
            lines.append(line)

    if not lines:
        raise RuntimeError("No session rows parsed from eval output.")

    # Convert fixed-width-ish text to dataframe
    df = pd.read_fwf(pd.compat.StringIO("\n".join(lines)))
    return df

def main():
    merged = None

    for model in MODELS:
        pred_dir = REPO / "results" / "infer" / "010626" / model
        print(f"Processing {model} ...")

        df = run_eval_to_df(pred_dir)

        df = df[["sheet", "Interval_P", "Interval_R", "Interval_F1"]]
        df = df.rename(columns={
            "Interval_P": f"{model}_P",
            "Interval_R": f"{model}_R",
            "Interval_F1": f"{model}_F1",
        })

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="sheet", how="outer")

    # Best model per session (by Interval_R)
    recall_cols = [c for c in merged.columns if c.endswith("_R")]
    merged["best_model_by_R"] = merged[recall_cols].idxmax(axis=1)

    out_path = REPO / "results" / "infer" / "010626" / "compare_3models_interval_table.xlsx"
    merged.to_excel(out_path, index=False)

    print("\n================ FINAL COMPARISON TABLE ================")
    print(merged.to_string(index=False))
    print(f"\n[OK] Saved table to: {out_path}")

if __name__ == "__main__":
    main()
