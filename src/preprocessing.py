import os
from glob import glob

# ==============================
# Configuration
# ==============================
CONFIG = {
    "INPUT_DIR": os.path.abspath(os.path.expanduser("~/gcs/inputs")),
    "MODE": "check",     # "list" or "check"
    "REQUIRED_VIEWS": ["UP", "SIDE1", "SIDE2"],
    
}


# ==============================
# Helper Functions
# ==============================
def find_all_videos():
    return sorted(glob(os.path.join(CONFIG["INPUT_DIR"], "*.mp4")))


def parse_session_and_view(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    parts = name.split(" ", 1)
    if len(parts) < 2:
        return None, None

    rest = parts[1]
    rest_parts = rest.split()
    if len(rest_parts) < 2:
        return None, None

    session_id = rest_parts[0]
    mouse_part, _, suffix = rest_parts[1].partition("-")
    session_name = f"{session_id} {mouse_part}"

    s = suffix.lower()
    if "webcamup" in s:
        view = "UP"
    elif "webcamside1" in s:
        view = "SIDE1"
    elif "webcamside2" in s:
        view = "SIDE2"
    else:
        view = "UNKNOWN"

    return session_name, view


def build_session_map(video_paths):
    sessions = {}
    for path in video_paths:
        session, view = parse_session_and_view(path)
        if not session:
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
