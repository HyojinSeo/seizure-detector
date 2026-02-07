#!/usr/bin/env bash
set -euo pipefail

cd ~/seizure-detector

MODEL="results/latefusion/all3_memmap_bs16_fixlabel/best_model.keras"
OUTDIR="results/infer"
BATCH=16
RAW_DIR="$HOME/gcs/inputs"

# Usage: ./scripts/run_infer_by_date.sh 010626
DATE="${1:-}"
if [[ -z "$DATE" ]]; then
    echo "Usage: $0 <MMDDYY>   (example: 010626)"
    exit 1
fi

# Thresholds to evaluate
THRS=(0.50 0.30)

mkdir -p "$OUTDIR"

# Collect unique sessions from filenames
# Example: POST KA010626 F1-webcamup.mp4 -> "KA010626 F1"
mapfile -t SESSIONS < <(
    ls -1 "$RAW_DIR" \
        | grep -i "POST KA${DATE}" \
	| sed -E 's/^POST (KA[0-9]{6}) ([A-Za-z0-9]+).*/\1 \2/I' \
	| sort -u
)

if [[ "${#SESSIONS[@]}" -eq 0 ]]; then
    echo "[ERROR] No sessions found for date KA${DATE} in $RAW_DIR"
    exit 2
fi

echo "[INFO] Found sessions:"
printf '  - %s\n' "${SESSIONS[@]}"

for thr in "${THRS[@]}"; do
      thr_tag=$(printf "thr%02d" "$(python - <<PY
t=float("$thr")
print(int(round(t*100)))
PY
)")
      for s in "${SESSIONS[@]}"; do
	  # "KA010626 F1" -> "KA010626_F1"
	  tag=$(echo "$s" | awk '{print $1"_"$2}')
	  out_xlsx="${OUTDIR}/${tag}_${thr_tag}_intervals.xlsx"

	  echo "[RUN] session='$s' threshold=$thr -> $out_xlsx"
	  python -m src.infer_timeline_latefusion \
		 --session "$s" \
		 --raw_video_dir "$RAW_DIR" \
		 --model_path "$MODEL" \
		 --out_xlsx "$out_xlsx" \
		 --threshold "$thr" \
		 --batch_size "$BATCH" || {
	      echo "[WARN] Failed session='$s' threshold=$thr. Skipping."
	      continue
	  }
      done
done

echo "[OK] All inference finished. Results in $OUTDIR"
