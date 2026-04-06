#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    repo = Path.home() / "seizure-detector"
    infer_py = repo / "src" / "infer_timeline_latefusion_dynamic.py"
    eval_py = repo / "scripts" / "eval_010626_pred_vs_gt.py"
    gt_xlsx = repo / "data" / "seizure_stage.xlsx"

    if not infer_py.exists():
        raise FileNotFoundError(f"Missing: {infer_py}")
    if not eval_py.exists():
        raise FileNotFoundError(f"Missing: {eval_py}")
    if not gt_xlsx.exists():
        raise FileNotFoundError(f"Missing: {gt_xlsx}")

    # 모델 3개 (exp_name, seq_len, stride)
    models = [
        ("all3_dyn_16_4", 16, 4),
        ("all3_dyn_4_2",   4, 2),
        ("all3_dyn_8_4",   8, 4),
    ]

    sessions = [
        "KA010626 F1",
        "KA010626 F1 B",
        "KA010626 F2",
        "KA010626 F2 B",
        "KA010626 M1",
        "KA010626 M1 B",
        "KA010626 M2",
        "KA010626 M2 B",
    ]

    threshold = 0.50
    batch = 16
    min_overlap_sec = 1.0
    step_sec = 1

    # 1) Inference per model
    for exp, seq_len, stride in models:
        model_path = repo / "results" / "latefusion" / exp / "best_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        out_dir = repo / "results" / "infer" / "010626" / exp
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== RUN INFER: {exp} (seq_len={seq_len}, stride={stride}) ====================")
        for sess in sessions:
            tag = sess.replace("KA", "").replace(" ", "_").replace("-", "_")
            out_xlsx = out_dir / f"{tag}_pred.xlsx"
            run([
                sys.executable, str(infer_py),
                "--session", sess,
                "--model_path", str(model_path),
                "--threshold", f"{threshold:.2f}",
                "--batch_size", str(batch),
                "--seq_len", str(seq_len),
                "--stride", str(stride),
                "--out_xlsx", str(out_xlsx),
            ])

    # 2) Eval per model
    for exp, _, _ in models:
        pred_dir = repo / "results" / "infer" / "010626" / exp
        print(f"\n==================== EVAL: {exp} ====================")
        run([
            sys.executable, str(eval_py),
            "--gt_xlsx", str(gt_xlsx),
            "--pred_dir", str(pred_dir),
            "--min_overlap_sec", str(min_overlap_sec),
            "--step_sec", str(step_sec),
        ])

    print("\n[OK] Done. Predictions saved under results/infer/010626/<exp_name>/")

if __name__ == "__main__":
    main()
