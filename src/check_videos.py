#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from glob import glob
from typing import Dict, Tuple, Optional

# ==============================
# Configuration
# ==============================
CONFIG = {
    "INPUT_DIR": os.path.abspath(os.path.expanduser("~/gcs/inputs")),
    "MODE": "check",  # "list" or "check"
    "REQUIRED_VIEWS": ["UP", "SIDE1", "SIDE2"],

    # Exclude sessions by date (KA010626 etc.)
    # Put date strings WITHOUT "KA" prefix.
    "EXCLUDE_DATES": {"010626"},
}

# ==============================
# Helper Functions
# ==============================
def find_all_videos():
    return sorted(glob(os.path.join(CONFIG["INPUT_DIR"], "*.mp4")))


def parse_session_and_view(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse:
      POST KA121325 M3 B-1-webcamup.mp4 -> ("121325 M3_B-1", "UP")
      POST KA121325 M3 B-webcamside1.mp4 -> ("121325 M3_B", "SIDE1")
      POST KA121325 M3-webcamside2.mp4 -> ("121325 M3", "SIDE2")

    Returns (session_name, view) or (None, None) if not matching.
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    low = name.lower()

    if "post ka" not in low:
        return None, None

    # Match date + animal + optional booster:
    # - date: 6 digits
    # - animal: M1/F2 etc
    # - booster: "B" or "B-1" or "B_2"
    #
    # Examples matched:
    #   POST KA121325 M3 B-1-webcamup
    #   POST KA121325 M3 B-webcamside1
    #   POST KA121325 M3-webcamside2
    m = re.search(r"post\s*ka(\d{6})\s*([mf]\d)\s*(b(?:[-_]?(\d+))?)?", low, re.IGNORECASE)
    if not m:
        return None, None

    date = m.group(1)              # "121325"
    animal = m.group(2).upper()    # "M3"
    b_full = m.group(3)            # "b", "b-1", "b_2", or None
    b_num = m.group(4)             # "1", "2", or None

    # Exclude unwanted dates (e.g., 010626)
    if date in CONFIG.get("EXCLUDE_DATES", set()):
        return None, None

    # Build session name
    if b_full:
        if b_num:
            session_name = f"{date} {animal}_B-{b_num}"
        else:
            session_name = f"{date} {animal}_B"
    else:
        session_name = f"{date} {animal}"

    # Determine view from webcam substring
    if "webcamup" in low:
        view = "UP"
    elif "webcamside1" in low:
        view = "SIDE1"
    elif "webcamside2" in low:
        view = "SIDE2"
    else:
        view = "UNKNOWN"

    return session_name, view


def build_session_map(video_paths):
    sessions: Dict[str, Dict[str, str]] = {}
    for path in video_paths:
        session, view = parse_session_and_view(path)
        if not session or not view:
            continue
        sessions.setdefault(session, {})[view] = path
    return sessions


def list_sessions(session_map):
    for session in sorted(session_map.keys()):
        print("•", session)


def check_views(session_map):
    req = CONFIG["REQUIRED_VIEWS"]
    for session in sorted(session_map.keys()):
        views = session_map[session].keys()
        missing = [v for v in req if v not in views]
        if not missing:
            print(f"[OK] {session}")
        else:
            print(f"[MISSING] {session} -> {', '.join(missing)}")


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    videos = find_all_videos()
    session_map = build_session_map(videos)

    if CONFIG["MODE"] == "list":
        list_sessions(session_map)
    elif CONFIG["MODE"] == "check":
        check_views(session_map)
