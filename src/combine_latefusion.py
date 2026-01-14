#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine per-session per-view sequences into a late-fusion dataset (MEMMAP version).

Input (from preprocess_seq_cnn_lstm.py):
  data/processed_seq/sessions/
    X_SEQ_TOP_<TOKEN>.npy,  y_SEQ_TOP_<TOKEN>.npy
    X_SEQ_SIDE_<TOKEN>.npy, y_SEQ_SIDE_<TOKEN>.npy
    X_SEQ_SIDE2_<TOKEN>.npy,y_SEQ_SIDE2_<TOKEN>.npy

Output:
  data/processed_seq/latefusion/<exp_name>/
    X_TOP.npy    (N,T,H,W,1) if TOP included
    X_SIDE.npy   (N,T,H,W,1) if SIDE included
    X_SIDE2.npy  (N,T,H,W,1) if SIDE2 included
    y.npy        (N,)
    index.csv
    meta.json

Examples:
  python -m src.combine_latefusion --exp_name top_only --views TOP
  python -m src.combine_latefusion --exp_name top_side --views TOP SIDE
  python -m src.combine_latefusion --exp_name all3 --views TOP SIDE SIDE2 --allow_trim_to_min
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.lib.format import open_memmap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESS_DIR = PROJECT_ROOT / "data" / "processed_seq" / "sessions"
OUT_ROOT = PROJECT_ROOT / "data" / "processed_seq" / "latefusion"


def list_tokens(sess_dir: Path) -> List[str]:
    tokens = set()
    # Token discovery uses TOP by default (same as your original).
    for p in sess_dir.glob("X_SEQ_TOP_*.npy"):
        m = re.match(r"X_SEQ_TOP_(.+)\.npy$", p.name)
        if m:
            tokens.add(m.group(1))
    return sorted(tokens)


def load_view_mmap(sess_dir: Path, view: str, token: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = sess_dir / f"X_SEQ_{view}_{token}.npy"
    y_path = sess_dir / f"y_SEQ_{view}_{token}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {view} files for {token}: {x_path.name}, {y_path.name}")

    # mmap_mode='r' avoids loading whole array into RAM
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r").reshape(-1)

    # Ensure X is (N,T,H,W,1)
    if X.ndim == 4:
        # (N,T,H,W) -> (N,T,H,W,1)
        # This creates a view; later we materialize per-session slice anyway.
        X = X[..., np.newaxis]
    if X.ndim != 5:
        raise ValueError(f"Unexpected X shape for {view} {token}: {X.shape}")

    return X, np.array(y, dtype=np.int64)  # y is small; load into RAM safely


def should_scale_0_1(X: np.ndarray) -> bool:
    """
    Decide whether to scale 0..255 -> 0..1.
    We avoid scanning entire X; sample a tiny subset.
    """
    try:
        if X.dtype == np.uint8:
            return True
        # sample up to first 8 samples only
        n = int(X.shape[0])
        k = min(8, n)
        if k <= 0:
            return False
        sample = np.array(X[:k], dtype=np.float32)
        mx = float(np.nanmax(sample))
        return mx > 1.5
    except Exception:
        return False


def write_index_header(index_csv_path: Path) -> None:
    with index_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["global_row", "session", "local_row", "label"])
        w.writeheader()


def append_index_rows(index_csv_path: Path, start_global: int, token: str, y: np.ndarray) -> None:
    with index_csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["global_row", "session", "local_row", "label"])
        for i, lab in enumerate(y.tolist()):
            w.writerow(
                {
                    "global_row": start_global + i,
                    "session": token,
                    "local_row": i,
                    "label": int(lab),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True, type=str)
    ap.add_argument("--views", nargs="+", required=True, choices=["TOP", "SIDE", "SIDE2"])
    ap.add_argument("--include", nargs="*", default=None, help="Optional tokens list: 112625F1 112625F1_B ...")
    ap.add_argument(
        "--allow_trim_to_min",
        action="store_true",
        help="If N differs across views for a session, trim to min N instead of crashing.",
    )
    args = ap.parse_args()

    if not SESS_DIR.exists():
        raise FileNotFoundError(f"Sessions dir not found: {SESS_DIR}")

    tokens = list_tokens(SESS_DIR)
    if args.include:
        allow = set(args.include)
        tokens = [t for t in tokens if t in allow]
    if not tokens:
        raise RuntimeError("No session tokens found to combine. Check SESS_DIR or --include.")

    views: List[str] = args.views

    out_dir = OUT_ROOT / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # PASS 1: determine N_total and shapes
    # --------------------------
    session_plan: List[Dict[str, object]] = []
    N_total = 0
    ref_T = ref_H = ref_W = ref_C = None
    scale_flags: Dict[str, bool] = {}

    for token in tokens:
        Xv: Dict[str, np.ndarray] = {}
        yv: Dict[str, np.ndarray] = {}

        for v in views:
            Xv[v], yv[v] = load_view_mmap(SESS_DIR, v, token)

        # label consistency across views
        y0 = yv[views[0]]
        for v in views[1:]:
            if y0.shape != yv[v].shape or not np.array_equal(y0, yv[v]):
                raise ValueError(f"Label mismatch across views for session {token}: {views[0]} vs {v}")

        # N consistency across views
        Ns = {v: int(Xv[v].shape[0]) for v in views}
        n_use = None
        if len(set(Ns.values())) == 1:
            n_use = list(Ns.values())[0]
        else:
            if not args.allow_trim_to_min:
                raise ValueError(f"N mismatch across views for session {token}: {Ns}")
            n_use = min(Ns.values())

        # shape consistency (T,H,W,C)
        _, T, H, W, C = Xv[views[0]].shape
        if ref_T is None:
            ref_T, ref_H, ref_W, ref_C = T, H, W, C
        else:
            if (T, H, W, C) != (ref_T, ref_H, ref_W, ref_C):
                raise ValueError(
                    f"Shape mismatch in {token}: got {(T,H,W,C)}, expected {(ref_T,ref_H,ref_W,ref_C)}"
                )

        # decide scaling per view (sample-based)
        for v in views:
            if v not in scale_flags:
                scale_flags[v] = should_scale_0_1(Xv[v])

        # store plan
        session_plan.append(
            {
                "token": token,
                "n_use": int(n_use),
                "pos": int(np.sum(y0[:n_use] == 1)),
            }
        )
        N_total += int(n_use)

        print(f"[PLAN] {token}: use N={int(n_use)}, pos={int(np.sum(y0[:n_use] == 1))}")

    if N_total <= 0:
        raise RuntimeError("Empty dataset after planning.")

    # --------------------------
    # Create memmap-backed .npy outputs
    # --------------------------
    # X_{view}.npy: float32
    X_out: Dict[str, np.ndarray] = {}
    for v in views:
        X_out[v] = open_memmap(
            filename=str(out_dir / f"X_{v}.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(N_total, ref_T, ref_H, ref_W, ref_C),
        )

    # y.npy: int64
    y_out = open_memmap(
        filename=str(out_dir / "y.npy"),
        mode="w+",
        dtype=np.int64,
        shape=(N_total,),
    )

    index_csv = out_dir / "index.csv"
    write_index_header(index_csv)

    # --------------------------
    # PASS 2: write data incrementally
    # --------------------------
    global_row = 0
    sessions_included: List[str] = []

    for item in session_plan:
        token = str(item["token"])
        n_use = int(item["n_use"])

        Xv: Dict[str, np.ndarray] = {}
        yv: Dict[str, np.ndarray] = {}
        for v in views:
            Xv[v], yv[v] = load_view_mmap(SESS_DIR, v, token)

        y0 = yv[views[0]][:n_use]
        y_out[global_row : global_row + n_use] = y0

        # index rows streamed to CSV (no big RAM list)
        append_index_rows(index_csv, global_row, token, y0)

        for v in views:
            Xi = Xv[v][:n_use]  # (N,T,H,W,1)
            # materialize as float32 slice for writing
            Xi_f = np.array(Xi, dtype=np.float32, copy=False)
            if scale_flags.get(v, False):
                Xi_f = Xi_f / 255.0
            X_out[v][global_row : global_row + n_use] = Xi_f

        sessions_included.append(token)

        print(f"[OK] {token}: wrote rows {global_row}..{global_row + n_use - 1} (N={n_use})")
        global_row += n_use

    # flush memmaps
    for v in views:
        X_out[v].flush()
    y_out.flush()

    meta = {
        "exp_name": args.exp_name,
        "views": views,
        "num_samples": int(N_total),
        "pos_samples": int(np.sum(np.array(y_out, dtype=np.int64) == 1)),
        "sessions_included": sessions_included,
        "scale_to_0_1": {v: bool(scale_flags.get(v, False)) for v in views},
    }
    for v in views:
        meta[f"shape_{v}"] = list(X_out[v].shape)
    meta["shape_y"] = list(y_out.shape)

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Combine complete (memmap). Outputs:")
    for v in views:
        print(f"  - {out_dir / f'X_{v}.npy'}  shape={tuple(X_out[v].shape)}")
    print(f"  - {out_dir / 'y.npy'}      shape={tuple(y_out.shape)}")
    print(f"  - {index_csv}")
    print(f"  - {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
