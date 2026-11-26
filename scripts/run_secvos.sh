#!/usr/bin/env bash
# Dual-Agent evaluation on SeCVOS.
#
#   SECVOS_ROOT        SeCVOS root
#   ASSESSOR_MODEL     Qwen3-VL-8B-Thinking
#   CORRECTOR_ADAPTER  corrector LoRA
#   OUTPUT_DIR         output directory
#   NUM_GPUS           number of GPUs (default 1)
#   VIDEO_IDS_FILE     optional list of video ids
set -euo pipefail
: "${SECVOS_ROOT:?set SECVOS_ROOT}"
: "${ASSESSOR_MODEL:?set ASSESSOR_MODEL}"
: "${CORRECTOR_ADAPTER:?set CORRECTOR_ADAPTER}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS:-1}"
PY_LAUNCH="${PY_LAUNCH:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA=()
if [ -n "${VIDEO_IDS_FILE:-}" ]; then EXTRA+=(--video_ids_file "$VIDEO_IDS_FILE"); fi

mkdir -p "$OUTPUT_DIR"
# shellcheck disable=SC2086
exec $PY_LAUNCH "$HERE/dual_agent/run_dual_agent.py" \
  --secvos_root "$SECVOS_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --assessor_model_path "$ASSESSOR_MODEL" \
  --corrector_model_path "$CORRECTOR_ADAPTER" \
  --assessor_format_prompt tracking_assessor_rich_format_prompt.txt \
  --corrector_format_prompt tracking_corrector_no_think_prompt.txt \
  --occlusion_poll_interval 12 \
  --reappearance_cooldown 8 \
  --auto_trigger_min_spacing 8 \
  --num_first_appearance 2 \
  --num_uniform 2 \
  --window_size 3 \
  --frame_stride 4 \
  --vlm_resize 1000 \
  --max_pixels 401408 \
  --vote_count 3 \
  --vote_stride 4 \
  --max_correction_rounds 3 \
  --correction_iou_threshold 0.5 \
  --assessor_max_eff_iou 0.7 \
  --neg_prompt_max_eff_iou 0.7 \
  --seed_sampling \
  --save_frame_details \
  --num_gpus "$NUM_GPUS" \
  ${EXTRA[@]+"${EXTRA[@]}"}
