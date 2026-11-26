#!/usr/bin/env bash
# Official LVOS evaluation.
#
#   LVOS_EVAL  lvos-evaluation repository
#   LVOS_ROOT  LVOS-v2 valid directory
#   PRED_DIR   predicted masks
set -euo pipefail
: "${LVOS_EVAL:?}" "${LVOS_ROOT:?}" "${PRED_DIR:?}"
cd "$LVOS_EVAL"
"${PYTHON:-python}" evaluation_method.py --task semi-supervised --set valid \
  --lvos_path "$LVOS_ROOT" --results_path "$PRED_DIR" --mp_nums "${NPROC:-16}"
