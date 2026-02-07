import re
import sys
from pathlib import Path

import pandas as pd

# ======================
# Helpers
# ======================

TIME_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)

def time_to_seconds(t: str) -> int | None:
    """
    Convert "mm:ss" or "h:mm:ss" to seconds.
    Returns None if format is invalid.
    """
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 2:
        mm, ss = parts
        if not (mm.isdigit() and ss.isdigit()):
            return None
        m = int(mm); s = int(ss)
        if s >= 60:
            return None
        return m * 60 + s
    elif len(parts) == 3:
        hh, mm, ss = parts
        if not (hh.isdigit() and mm.isdigit() and ss.isdigit()):
            return None
        h = int(hh); m = int(mm); s = int(ss)
        if m >= 60 or s >= 60:
            return None
        return h * 3600 + m * 60 + s
    return None

def normalize_header(values) -> list[str]:
    # Convert to string, strip whitespace; keep exact text matching after strip
    out = []
    for v in values:
        if v is None:
            out.append("")
        else:
            out.append(str(v).strip())
    return out

def is_sheet_effectively_empty(df: pd.DataFrame) -> bool:
    # Sheet is empty if dataframe has no rows/cols OR all cells are NaN/blank strings
    if df is None:
        return True
    if df.shape[0] == 0 and df.shape[1] == 0:
        return True
    if df.shape[0] == 0:
        return True
    # If all values are NaN or blank after stripping -> empty
    tmp = df.copy()
    tmp = tmp.applymap(lambda x: "" if pd.isna(x) else str(x).strip())
    return (tmp.values == "").all()

# ======================
# Main check function
# ======================

def check_excel(excel_path: str) -> pd.DataFrame:
    excel_path = str(Path(excel_path).expanduser())

    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    sheet_names = xls.sheet_names

    results = []

    print(f"Found {len(sheet_names)} sheets in: {excel_path}\n")

    for sheet in sheet_names:
        print(f"Checking sheet: {sheet}")

        # Read raw (no header) to validate the first row exactly
        df_raw = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            header=None,
            dtype=str,
            engine="openpyxl",
        )

        status = {
            "sheet": sheet,
            "is_empty": False,
            "header_ok": False,
            "time_ok": False,
            "issues": [],
            "bad_time_rows": 0,
            "bad_time_examples": "",
        }

        if is_sheet_effectively_empty(df_raw):
            status["is_empty"] = True
            status["header_ok"] = True  # not applicable
            status["time_ok"] = True    # not applicable
            print("  - Sheet is empty.\n")
            results.append(status)
            continue

        # Header check: first row must be exactly Time, Description, Stage
        header = normalize_header(df_raw.iloc[0, :3].tolist())
        expected = ["Time", "Description", "Stage"]
        if header == expected:
            status["header_ok"] = True
        else:
            status["issues"].append(f"Header mismatch: got {header}, expected {expected}")
            status["header_ok"] = False

        # If header ok, validate time column values (all non-empty rows)
        # We will look at column 0 from row 1 onward.
        bad_examples = []
        bad_count = 0

        # Only attempt time checks if we have at least 2 rows (header + data)
        if df_raw.shape[0] >= 2:
            time_col = df_raw.iloc[1:, 0]  # below header, first column
            for idx, val in time_col.items():  # idx is actual row index in df_raw
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if s == "":
                    continue

                m = TIME_RANGE_RE.match(s)
                if not m:
                    bad_count += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"Row {idx+1}: '{s}' (format)")
                    continue

                start_str = m.group("start")
                end_str = m.group("end")
                start_sec = time_to_seconds(start_str)
                end_sec = time_to_seconds(end_str)

                if start_sec is None or end_sec is None:
                    bad_count += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"Row {idx+1}: '{s}' (invalid time)")
                    continue

                if start_sec > end_sec:
                    bad_count += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"Row {idx+1}: '{s}' (start > end)")
                    continue

        # Decide time_ok
        if bad_count == 0:
            status["time_ok"] = True
        else:
            status["time_ok"] = False
            status["bad_time_rows"] = bad_count
            status["bad_time_examples"] = " | ".join(bad_examples)
            status["issues"].append(f"Bad time rows: {bad_count}")

        # Print quick per-sheet summary
        if status["is_empty"]:
            print("  - Empty sheet.\n")
        else:
            if status["header_ok"] and status["time_ok"]:
                print("  - OK (header and time checks passed).\n")
            else:
                if not status["header_ok"]:
                    print("  - Header check FAILED.")
                    print(f"    Expected: {expected}")
                    print(f"    Got:      {header}")
                if not status["time_ok"]:
                    print(f"  - Time check FAILED. Bad rows: {status['bad_time_rows']}")
                    if status["bad_time_examples"]:
                        print(f"    Examples: {status['bad_time_examples']}")
                print()

        results.append(status)

    summary = pd.DataFrame(results)

    # Derive a simple classification label
    def classify(row):
        if row["is_empty"]:
            return "EMPTY"
        if not row["header_ok"]:
            return "BAD_HEADER"
        if not row["time_ok"]:
            return "BAD_TIME"
        return "OK"

    summary["status"] = summary.apply(classify, axis=1)

    # A compact view at the end
    summary_view = summary[[
        "sheet", "status", "is_empty", "header_ok", "time_ok", "bad_time_rows", "bad_time_examples", "issues"
    ]].copy()

    return summary_view

# ======================
# CLI entrypoint
# ======================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_excel_sheets.py /path/to/your.xlsx")
        sys.exit(1)

    excel_file = sys.argv[1]
    df_summary = check_excel(excel_file)

    print("\n======================")
    print("Final Summary Table")
    print("======================")
    # Print a readable table
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 120):
        print(df_summary.to_string(index=False))

    # Save summary next to the Excel
    out_path = Path(excel_file).with_suffix("")  # drop .xlsx
    out_csv = str(out_path) + "_sheet_check_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV: {out_csv}")
