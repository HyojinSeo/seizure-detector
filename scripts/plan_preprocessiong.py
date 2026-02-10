#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

OUTPUT_DIR = Path("data/processed_seq/sessions")

VIEWS = ["TOP", "SIDE", "SIDE2"]

TARGET_SESSIONS = [
    # 112625
    "112625 M1", "112625 M1_B",
    "112625 M2", "112625 M2_B",
    "112625 F1", "112625 F1_B",
    "112725 F2", "112725 F2_B",

    # 120725
    "120725 F1", "120725 F1_B",
    "120725 F2", "120725 F2_B",
    "120725 M1", "120725 M1_B",

    # 121125
    "121125 F1", "121125 F1_B",
    "121125 M1", "121125 M1_B",
    "121125 M2", "121125 M2_B",

    # 121225
    "121225 F1", "121225 F1_B",
    "121225 F2", "121225 F2_B",
    "121225 M1", "121225 M1_B",

    # 121325
    "121325 M1", "121325 M1_B",
    "121325 M2", "121325 M2_B",
    "121325 M3", "121325 M3_B-1",
]

def norm(session_id: str) -> str:
    return session_id.replace(" ", "")

def is_done(session_id: str) -> bool:
    s = norm(session_id)
    for v in VIEWS:
        x = OUTPUT_DIR / f"X_SEQ_{v}_{s}.npy"
        y = OUTPUT_DIR / f"y_SEQ_{v}_{s}.npy"
        if not x.exists() or not y.exists():
            return False
    return True

done = []
todo = []
missing_outputdir = not OUTPUT_DIR.exists()

if missing_outputdir:
    print(f"[ERROR] Output dir not found: {OUTPUT_DIR}")
    raise SystemExit(1)

for sid in TARGET_SESSIONS:
    (done if is_done(sid) else todo).append(sid)

print("\n=== DONE (already preprocessed) ===")
for s in done:
    print(s)

print("\n=== TODO (need preprocessing) ===")
for s in todo:
    print(s)

if todo:
    env_value = ",".join(todo)
    print("\nRun this to preprocess only TODO sessions:")
    print(f'SESSIONS="{env_value}" python -m src.preprocess_seq_cnn_lstm')
