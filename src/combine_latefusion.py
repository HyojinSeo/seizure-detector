#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine per-session per-view sequences into a late-fusion dataset.

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
  python -m src.combine_latefusion --exp_name all3 --views TOP SIDE SIDE2
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESS_DIR = PROJECT_ROOT / "data" / "processed_seq" / "sessions"
OUT_ROOT = PROJECT_ROOT / "data" / "processed_seq" / "latefusion"


def list_tokens(sess_dir: Path) -> List[str]:
    tokens = set()
    # Prefer TOP for token discovery; if you sometimes have SIDE-only sessions,
    # you can extend this to scan SIDE as well.
    for p in sess_dir.glob("X_SEQ_TOP_*.npy"):
        m = re.match(r"X_SEQ_TOP_(.+)\.npy$", p.name)
        if m:
            tokens.add(m.group(1))
    return sorted(tokens)


def load_view(sess_dir: Path, view: str, token: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = sess_dir / f"X_SEQ_{view}_{token}.npy"
    y_path = sess_dir / f"y_SEQ_{view}_{token}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {view} files for {token}: {x_path.name}, {y_path.name}")
    X = np.load(x_path)
    y = np.load(y_path).reshape(-1)
    if X.ndim == 4:
        X = X[..., np.newaxis]  # (N,T,H,W,1)
    if X.ndim != 5:
        raise ValueError(f"Unexpected X shape for {view} {token}: {X.shape}")
    return X, y


def maybe_scale_0_1(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    # If values look like 0..255, scale to 0..1
    if np.nanmax(X) > 1.5:
        X = X / 255.0
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True, type=str)
    ap.add_argument("--views", nargs="+", required=True, choices=["TOP", "SIDE", "SIDE2"])
    ap.add_argument("--include", nargs="*", default=None, help="Optional tokens list: 112625F1 112625F1_B ...")
    ap.add_argument("--allow_trim_to_min", action="store_true",
                    help="If N differs across views for a session, trim to min N instead of crashing.")
    args = ap.parse_args()

    if not SESS_DIR.exists():
        raise FileNotFoundError(f"Sessions dir not found: {SESS_DIR}")

    tokens = list_tokens(SESS_DIR)
    if args.include:
        allow = set(args.include)
        tokens = [t for t in tokens if t in allow]
    if not tokens:
        raise RuntimeError("No session tokens found to combine. Check SESS_DIR or --include.")

    out_dir = OUT_ROOT / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    views = args.views
    X_by_view: Dict[str, List[np.ndarray]] = {v: [] for v in views}
    y_all: List[np.ndarray] = []
    index_rows: List[Dict[str, object]] = []
    global_row = 0

    for token in tokens:
        Xv = {}
        yv = {}
        for v in views:
            Xv[v], yv[v] = load_view(SESS_DIR, v, token)

        # label consistency
        y0 = yv[views[0]]
        for v in views[1:]:
            if not np.array_equal(y0, yv[v]):
                raise ValueError(f"Label mismatch across views for session {token}: {views[0]} vs {v}")

        # N consistency
        Ns = {v: Xv[v].shape[0] for v in views}
        if len(set(Ns.values())) != 1:
            if not args.allow_trim_to_min:
                raise ValueError(f"N mismatch across views for session {token}: {Ns}")
            nmin = min(Ns.values())
            for v in views:
                Xv[v] = Xv[v][:nmin]
            y0 = y0[:nmin]

        # append
        for v in views:
            X_by_view[v].append(maybe_scale_0_1(Xv[v]))
        y_all.append(y0.astype(np.int64))

        n = int(y0.shape[0])
        pos = int(np.sum(y0 == 1))
        print(f"[OK] {token}: N={n}, pos={pos}")

        for i in range(n):
            index_rows.append({"global_row": global_row + i, "session": token, "local_row": i, "label": int(y0[i])})
        global_row += n

    y = np.concatenate(y_all, axis=0)
    meta = {
        "exp_name": args.exp_name,
        "views": views,
        "num_samples": int(y.shape[0]),
        "pos_samples": int(np.sum(y == 1)),
        "sessions_included": tokens,
    }

    for v in views:
        X = np.concatenate(X_by_view[v], axis=0)
        np.save(out_dir / f"X_{v}.npy", X)
        meta[f"shape_{v}"] = list(X.shape)
        print(f"Saved: {out_dir / f'X_{v}.npy'} shape={X.shape}")

    np.save(out_dir / "y.npy", y)
    pd.DataFrame(index_rows).to_csv(out_dir / "index.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_dir / 'y.npy'} shape={y.shape}")
    print(f"Saved: {out_dir / 'index.csv'}")
    print(f"Saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
