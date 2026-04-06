#!/usr/bin/env bash
set -euo pipefail

MODEL="results/latefusion/all3/best_model.keras"
GT_XLSX="data/seizure_stage.xlsx"
OUT_DIR="results/infer/010626/all3"
THRESH="0.50"
BATCH="16"

mkdir -p "$OUT_DIR"

# (session_arg_for_infer, gt_sheet_name, out_stub)
SESSIONS=(
  "KA010626 F1|010626F1|010626_F1"
  "KA010626 F1 B|010626F1_B|010626_F1_B"
  "KA010626 F2|010626F2|010626_F2"
  "KA010626 F2 B|010626F2_B|010626_F2_B"
  "KA010626 M1|010626M1|010626_M1"
  "KA010626 M1 B|010626M1_B|010626_M1_B"
  "KA010626 M2|010626M2|010626_M2"
  "KA010626 M2 B|010626M2_B|010626_M2_B"
)

for item in "${SESSIONS[@]}"; do
  IFS="|" read -r SESSION SHEET STUB <<< "$item"

  PRED_XLSX="${OUT_DIR}/${STUB}_pred.xlsx"
  OUT_PNG="${OUT_DIR}/${STUB}_timeline.png"

  echo "=================================================================="
  echo "[RUN] session=${SESSION} | sheet=${SHEET}"
  echo "  pred: ${PRED_XLSX}"
  echo "  png : ${OUT_PNG}"

  # 1) inference -> pred xlsx
  python src/infer_timeline_latefusion.py \
    --session "${SESSION}" \
    --model_path "${MODEL}" \
    --threshold "${THRESH}" \
    --batch_size "${BATCH}" \
    --out_xlsx "${PRED_XLSX}"

  # 2) plot GT(sheet) vs pred
  python scripts/plot_timeline_label_vs_pred.py \
    --label_xlsx "${GT_XLSX}" \
    --label_sheet "${SHEET}" \
    --pred_xlsx "${PRED_XLSX}" \
    --out_png "${OUT_PNG}" \
    --no_show
done

echo "=================================================================="
echo "[OK] Done. Outputs in: ${OUT_DIR}"
