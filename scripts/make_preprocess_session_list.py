import re
from pathlib import Path
import pandas as pd

# ======================
# Configuration
# ======================

# Directory containing input videos (SIDE1/SIDE2/TOP)
VIDEO_DIR = Path("/home/kimlabmouse/gcs/inputs")

# Excel file containing per-session labeling sheets
EXCEL_PATH = Path.home() / "Input.xlsx"

# Session keys to exclude explicitly
EXCLUDE_KEYS = {"121325M3_B2"}

# Match filenames like:
# "POST KA112625 M1 B-webcamside1.mp4"
# "POST KA121325 M3 B2-webcamup.mp4"
VIDEO_PATTERN = re.compile(
    r"POST KA(?P<date>\d{6})\s+(?P<mouse>[FM]\d)\s*(?P<variant>B1|B2|B)?-webcam(?P<cam>side1|side2|up)\.mp4$",
    re.IGNORECASE,
)

# Video extensions to consider
VIDEO_EXTS = {".mp4"}

def make_key(date: str, mouse: str, variant: str | None) -> str:
    # Build a session key like "112625F1" or "112625F1_B" or "121325M3_B2"
    if variant:
        return f"{date}{mouse}_{variant}"
    return f"{date}{mouse}"

def scan_videos(video_dir: Path) -> dict[str, set[str]]:
    # Map session key -> set of camera views found (side1, side2, up)
    key_to_cams: dict[str, set[str]] = {}

    for p in video_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue

        m = VIDEO_PATTERN.match(p.name)
        if not m:
            continue

        date = m.group("date")
        mouse = m.group("mouse")
        variant = m.group("variant")
        cam = m.group("cam").lower()

        key = make_key(date, mouse, variant)
        key_to_cams.setdefault(key, set()).add(cam)

    return key_to_cams

def sheet_is_empty(df_raw: pd.DataFrame) -> bool:
    # Treat a sheet as empty if all cells are blank/NaN after stripping
    if df_raw is None or df_raw.shape[0] == 0:
        return True
    tmp = df_raw.copy()
    tmp = tmp.applymap(lambda x: "" if pd.isna(x) else str(x).strip())
    return (tmp.values == "").all()

def nonempty_sheets(excel_path: Path) -> set[str]:
    # Return set of sheet names that are not empty
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    keep: set[str] = set()
    for s in xls.sheet_names:
        df_raw = pd.read_excel(excel_path, sheet_name=s, header=None, dtype=str, engine="openpyxl")
        if not sheet_is_empty(df_raw):
            keep.add(s.strip())
    return keep

def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"VIDEO_DIR not found: {VIDEO_DIR}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"EXCEL_PATH not found: {EXCEL_PATH}")

    key_to_cams = scan_videos(VIDEO_DIR)

    # Keep only sessions with all three views present
    three_view = {k for k, cams in key_to_cams.items() if {"side1", "side2", "up"}.issubset(cams)}

    # Keep only sessions whose Excel sheets are non-empty
    sheets_keep = nonempty_sheets(EXCEL_PATH)

    # Final include list: 3-view + non-empty sheet - explicit excludes
    include = sorted((three_view & sheets_keep) - EXCLUDE_KEYS)

    # Diagnostics
    nonempty_missing_videos = sorted(sheets_keep - three_view)
    threeview_missing_or_empty_sheet = sorted(three_view - sheets_keep)

    print("=== Include sessions (3-view + non-empty sheet, excluding requested keys) ===")
    for k in include:
        print(k)

    print("\n=== Non-empty sheets but missing 3-view videos ===")
    for k in nonempty_missing_videos:
        print(k)

    print("\n=== 3-view video sessions but sheet missing (or sheet empty) ===")
    for k in threeview_missing_or_empty_sheet:
        print(k)

    # Save include list for the preprocessing step
    out = Path("preprocess_include_sessions.txt")
    out.write_text("\n".join(include) + ("\n" if include else ""), encoding="utf-8")
    print(f"\nSaved include list: {out.resolve()}")

    # Save a CSV summary for quick review
    rows = []
    all_keys = sorted((sheets_keep | three_view) - EXCLUDE_KEYS)
    for k in all_keys:
        rows.append({
            "session_key": k,
            "has_3view_videos": k in three_view,
            "has_nonempty_sheet": k in sheets_keep,
            "included": k in include,
            "cams_found": ",".join(sorted(key_to_cams.get(k, set()))),
        })
    df = pd.DataFrame(rows)
    csv_out = Path("preprocess_include_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV: {csv_out.resolve()}")

if __name__ == "__main__":
    main()
