#!/usr/bin/env bash
# Official SeC evaluation (overall and per scene segment).
#
#   SEC_REPO     SeC repository
#   SECVOS_ROOT  SeCVOS root
#   PRED_DIR     predicted masks
set -euo pipefail
: "${SEC_REPO:?set SEC_REPO}"
: "${SECVOS_ROOT:?set SECVOS_ROOT}"
: "${PRED_DIR:?set PRED_DIR}"
PYTHON="${PYTHON:-python}"

n_gt=$(ls "$SECVOS_ROOT/Annotations" | wc -l)
n_pred=$(ls "$PRED_DIR" | grep -v -c '\.csv$' || true)
[ "$n_pred" = "$n_gt" ] || { echo "ERROR: $n_pred predicted videos vs $n_gt GT videos"; exit 1; }

cd "$SEC_REPO"
"$PYTHON" -m vos_evaluation.sav_evaluator \
  --gt_root "$SECVOS_ROOT/Annotations" \
  --pred_root "$PRED_DIR" \
  --scene_info "$SECVOS_ROOT/scene_info.json" \
  -s -n 8

for f in overall first_segment second_segment remain_segments; do
  echo "$f: $(grep Global "$PRED_DIR/results_$f.csv")"
done
