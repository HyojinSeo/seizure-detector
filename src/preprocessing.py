import os
from glob import glob

# ==============================
# Configuration
# ==============================
CONFIG = {
    "INPUT_DIR": "~/gcs/inputs",      # where videos are stored
    "VIDEO_EXT": ".mp4",              # file extension to search
}


# ==============================
# Helper Functions
# ==============================
def list_videos():
    input_dir = os.path.abspath(os.path.expanduser(CONFIG["INPUT_DIR"]))
    ext = CONFIG["VIDEO_EXT"]

    pattern = os.path.join(input_dir, f"*{ext}")
    videos = sorted(glob(pattern))

    print("Detected MP4 videos:")
    print("------------------------------------------------------------")
    if not videos:
        print(f"No {ext} files found in {input_dir}")
    else:
        for v in videos:
            print("•", v)

    return videos


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    list_videos()
