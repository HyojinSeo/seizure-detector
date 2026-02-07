#!/usr/bin/env bash
set -euo pipefail

cd ~/seizure-detector

PRED_DIR="results/infer"
LABEL_DIR="data/labels"
OUT_SUMMARY="${PRED_DIR}/010626_eval_summary.csv"

SESSIONS=("F1" "F1B" "F2" "F2B")

PRED_SUFFIXES=(
  "_intervals.xlsx"
  "_thr50_intervals.xlsx"
  "_thr30_intervals.xlsx"
)

label_path_for() {
  local s="$1"
  case "$s" in
    F1)  echo "${LABEL_DIR}/010626F1.xlsx" ;;
    F1B) echo "${LABEL_DIR}/010626F1_B.xlsx" ;;
    F2)  echo "${LABEL_DIR}/010626F2.xlsx" ;;
    F2B) echo "${LABEL_DIR}/010626F2_B.xlsx" ;;
    *)   echo "" ;;
  esac
}

echo "session,pred_file,tp,fp,fn,precision,recall" > "$OUT_SUMMARY"

for s in "${SESSIONS[@]}"; do
  label_xlsx="$(label_path_for "$s")"
  if [[ ! -f "$label_xlsx" ]]; then
    echo "[WARN] Missing label file: $label_xlsx (skipping $s)"
    continue
  fi

  for suf in "${PRED_SUFFIXES[@]}"; do
    pred_xlsx="${PRED_DIR}/KA010626_${s}${suf}"
    if [[ ! -f "$pred_xlsx" ]]; then
      echo "[WARN] Missing pred file: $pred_xlsx (skipping)"
      continue
    fi

    echo "[RUN] compare_intervals: session=$s pred=$(basename "$pred_xlsx")"
    out="$(
      python -m src.compare_intervals \
        --label_xlsx "$label_xlsx" \
        --pred_xlsx "$pred_xlsx" \
        2>&1
    )"
    echo "$out"

    tp=$(echo "$out" | grep -Eo 'TP:\s*[0-9]+' | awk '{print $2}' | head -n1 || true)
    fp=$(echo "$out" | grep -Eo 'FP:\s*[0-9]+' | awk '{print $2}' | head -n1 || true)
    fn=$(echo "$out" | grep -Eo 'FN:\s*[0-9]+' | awk '{print $2}' | head -n1 || true)
    precision=$(echo "$out" | grep -Eo 'Precision:\s*[0-9.]+' | awk '{print $2}' | head -n1 || true)
    recall=$(echo "$out" | grep -Eo 'Recall:\s*[0-9.]+' | awk '{print $2}' | head -n1 || true)

    echo "${s},$(basename "$pred_xlsx"),${tp},${fp},${fn},${precision},${recall}" >> "$OUT_SUMMARY"

    out_png="${pred_xlsx%.xlsx}_timeline.png"
    python -m src.plot_timeline \
      --label_xlsx "$label_xlsx" \
      --pred_xlsx "$pred_xlsx" \
      --out_png "$out_png"
  done
done

echo "[OK] Summary saved to $OUT_SUMMARY"
