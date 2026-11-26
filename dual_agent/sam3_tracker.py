"""SAM 3 video tracker: loading, per-frame readout and correction prompts."""

import logging
from typing import List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_tracker():
    from sam3.model_builder import build_sam3_video_model
    logger.info("Loading SAM 3")
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    tracker.eval()
    return tracker


def mask_to_box(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_logits(video_res_masks, index: int, height: int, width: int) -> np.ndarray:
    """Mask logits of one object at video resolution."""
    logits = (video_res_masks[index, 0] if video_res_masks.ndim == 4 else video_res_masks[index]).cpu().numpy()
    if logits.shape != (height, width):
        logits = cv2.resize(logits.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return logits


def objectness(obj_scores, index: int) -> float:
    if isinstance(obj_scores, torch.Tensor):
        flat = obj_scores.flatten()
        return float(flat[index].item() if index < flat.numel() else flat[0].item())
    try:
        return float(obj_scores[index])
    except Exception:
        return float(obj_scores)


def effective_iou(inference_state, obj_scores, frame_idx: int, index: int) -> Optional[float]:
    """SAM 3's confidence in its mask: max(2 * sigmoid(objectness) - 1, 0) * predicted IoU.
    None when the predicted IoU of this frame is not cached."""
    if obj_scores is None:
        return None
    consolidated = inference_state.get("consolidated_frame_inds", {})
    key = ("cond_frame_outputs" if frame_idx in consolidated.get("cond_frame_outputs", set())
           else "non_cond_frame_outputs")
    output = inference_state.get("output_dict", {}).get(key, {}).get(frame_idx)
    if output is None or "iou_score" not in output:
        return None
    if isinstance(obj_scores, torch.Tensor):
        score = obj_scores.flatten()[index]
    else:
        score = torch.tensor(float(obj_scores[index]) if hasattr(obj_scores, "__getitem__") else float(obj_scores))
    presence = (score.sigmoid() * 2 - 1).clamp(min=0)
    predicted_iou = output["iou_score"].float().flatten()[index]
    return float((presence * predicted_iou).item())


def _forget_frame(inference_state, frame_idx: int):
    """Drop the cached output of a frame so propagation recomputes it from the new prompt."""
    outputs = inference_state["output_dict"]
    for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        outputs[key].pop(frame_idx, None)
    for frames in inference_state.get("consolidated_frame_inds", {}).values():
        frames.discard(frame_idx)


def prompt_box(tracker, inference_state, frame_idx: int, obj_id: int, box: List[int], width: int, height: int):
    x1, y1, x2, y2 = box
    tracker.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], dtype=torch.float32),
        rel_coordinates=True,
        clear_old_points=True,
        use_prev_mem_frame=False,
    )
    _forget_frame(inference_state, frame_idx)


def prompt_absent(tracker, inference_state, frame_idx: int, obj_id: int, width: int, height: int):
    tracker.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=torch.zeros((height, width), dtype=torch.float32),
    )
    _forget_frame(inference_state, frame_idx)
