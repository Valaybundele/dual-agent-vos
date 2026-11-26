#!/usr/bin/env bash
# Dual-Agent evaluation on LVOS-v2 val.
#
#   LVOS_ROOT          LVOS-v2 valid directory
#   ASSESSOR_MODEL     Qwen3-VL-8B-Thinking
#   CORRECTOR_ADAPTER  corrector LoRA
#   OUTPUT_DIR         output directory
#   NUM_GPUS           number of GPUs (default 1)
#   VIDEO_IDS_FILE     optional list of video ids
set -euo pipefail
: "${LVOS_ROOT:?}" "${ASSESSOR_MODEL:?}" "${CORRECTOR_ADAPTER:?}" "${OUTPUT_DIR:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(--dataset lvos --data_root "$LVOS_ROOT" --output_dir "$OUTPUT_DIR"
      --assessor_model_path "$ASSESSOR_MODEL" --corrector_model_path "$CORRECTOR_ADAPTER"
      --no-streaming_trigger --num_gpus "${NUM_GPUS:-1}")
[ -n "${VIDEO_IDS_FILE:-}" ] && ARGS+=(--video_ids_file "$VIDEO_IDS_FILE")
# shellcheck disable=SC2086
exec ${PY_LAUNCH:-python} "$HERE/dual_agent/run.py" "${ARGS[@]}" ${EXTRA_ARGS:-}
