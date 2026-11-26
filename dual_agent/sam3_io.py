"""SeCVOS loading, SAM 3 helpers and mask output."""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

NAN = float("nan")



@dataclass
class FrameRecord:
    frame_idx: int
    sam3_bbox: Optional[List[int]]   # SAM 3 box (None when SAM 3 reports the object absent)
    vlm_bbox: Optional[List[int]]    # box after the Dual-Agent step (= sam3_bbox unless corrected)
    gt_bbox: Optional[List[int]]     # GT box (None when absent or not annotated)
    frame_annotated: bool            # this frame has a GT annotation for the object
    objectness: float                # SAM 3 objectness logit
    vlm_triggered: bool = False      # the assessor was called on this frame
    vlm_corrected: bool = False      # a correction was committed on this frame
    trigger_reason: str = ""         # "reappear" / "occ_poll" / "streaming_var"
    eff_iou: float = NAN
    mask_logit_mean: float = NAN
    mask_logit_margin: float = NAN
    mask_logit_global_max: float = NAN
    n_components: int = -1
    boundary_entropy: float = NAN
    energy_separation: float = NAN
    temporal_mask_iou: float = NAN
    pred_mask_area: int = 0
    is_occluded: bool = False
    occluded_frames: int = 0
    frames_since_reappearance: int = 0
    assessor_verdict: Optional[str] = None
    vote_verdicts: Optional[List[Optional[str]]] = None
    corrector_called: bool = False
    corrector_future_fidxs: Optional[List[int]] = None


@dataclass
class SequenceResult:
    sequence_id: str
    obj_id: int
    frames: List[FrameRecord] = field(default_factory=list)
    pred_masks: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)

    @property
    def vlm_calls(self):
        return sum(1 for fr in self.frames if fr.vlm_triggered)

    @property
    def vlm_corrections(self):
        return sum(1 for fr in self.frames if fr.vlm_corrected)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0



def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_annotations(masks_root: str, seq_id: str) -> Dict[int, Dict[int, Optional[List[int]]]]:
    """Return {frame_idx: {obj_id: bbox or None}} from per-object mask folders."""
    ann_dir = Path(masks_root) / seq_id
    gt: Dict[int, Dict[int, Optional[List[int]]]] = defaultdict(dict)
    if not ann_dir.exists():
        logger.warning(f"No annotations found for {seq_id}")
        return gt
    for obj_dir in sorted(ann_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        oid = int(obj_dir.name)
        for png in sorted(obj_dir.glob("*.png")):
            fidx = int(png.stem)
            mask = np.array(PILImage.open(png))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            gt[fidx][oid] = mask_to_bbox((mask > 0).astype(np.uint8))
    return gt


def get_gt_frame_indices(masks_root: str, seq_id: str) -> Dict[int, List[int]]:
    """Return {obj_id: [annotated frame indices]}."""
    ann_dir = Path(masks_root) / seq_id
    result: Dict[int, List[int]] = {}
    if not ann_dir.exists():
        return result
    for obj_dir in sorted(ann_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        result[int(obj_dir.name)] = sorted(int(p.stem) for p in obj_dir.glob("*.png"))
    return result


def make_mask_loader(masks_root: str, seq_id: str):
    """Return loader(frame_idx, obj_id) -> binary uint8 GT mask or None."""
    ann_dir = Path(masks_root) / seq_id

    def loader(frame_idx: int, obj_id: int) -> Optional[np.ndarray]:
        png = ann_dir / f"{obj_id:03d}" / f"{frame_idx:05d}.png"
        if not png.exists():
            return None
        m = np.array(PILImage.open(png))
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)

    return loader


def load_frame_paths(frames_root: str, seq_id: str) -> List[str]:
    return sorted(str(p) for p in (Path(frames_root) / seq_id).glob("*.jpg"))


def save_masks(seq_results: Dict[int, "SequenceResult"], output_dir: str, seq_id: str,
               video_h: int, video_w: int,
               gt_frame_indices: Optional[Dict[int, List[int]]] = None):
    """Write {output_dir}/{seq_id}/{obj_id:03d}/{frame_idx:05d}.png for every
    GT-annotated frame (empty mask where there is no prediction)."""
    for obj_id, res in seq_results.items():
        obj_dir = os.path.join(output_dir, seq_id, f"{obj_id:03d}")
        os.makedirs(obj_dir, exist_ok=True)
        if gt_frame_indices and obj_id in gt_frame_indices:
            frames_to_save = gt_frame_indices[obj_id]
        elif res.pred_masks:
            frames_to_save = sorted(res.pred_masks.keys())
        else:
            continue
        for fidx in frames_to_save:
            m = res.pred_masks.get(fidx)
            if m is None:
                m = np.zeros((video_h, video_w), dtype=np.uint8)
            elif m.shape != (video_h, video_w):
                m = cv2.resize(m.astype(np.uint8), (video_w, video_h),
                               interpolation=cv2.INTER_NEAREST)
            mask_img = PILImage.fromarray((m > 0).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.convert("1")
            mask_img.save(os.path.join(obj_dir, f"{fidx:05d}.png"))



def build_tracker():
    from sam3.model_builder import build_sam3_video_model
    logger.info("Loading SAM3 model...")
    m = build_sam3_video_model()
    tracker = m.tracker
    tracker.backbone = m.detector.backbone
    tracker.eval()
    logger.info("SAM3 loaded.")
    return tracker


def pred_to_bbox(video_res_masks, obj_idx: int, h: int, w: int) -> Optional[List[int]]:
    if video_res_masks is None:
        return None
    mask = (video_res_masks[obj_idx, 0] if video_res_masks.ndim == 4
            else video_res_masks[obj_idx]).cpu().numpy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return mask_to_bbox((mask > 0).astype(np.uint8))


def objectness(obj_scores, obj_idx: int) -> float:
    if isinstance(obj_scores, torch.Tensor):
        flat = obj_scores.flatten()
        return float(flat[obj_idx].item() if obj_idx < flat.numel() else flat[0].item())
    try:
        return float(obj_scores[obj_idx])
    except Exception:
        return float(obj_scores)


def eff_iou_score(inference_state, obj_scores, frame_idx: int, obj_idx: int) -> Optional[float]:
    """SAM 3 self-confidence: max(2*sigmoid(objectness) - 1, 0) * predicted mask IoU.
    None when the predicted IoU for this frame is not cached."""
    if obj_scores is None:
        return None
    output_dict = inference_state.get("output_dict", {})
    consolidated = inference_state.get("consolidated_frame_inds", {})
    if frame_idx in consolidated.get("cond_frame_outputs", set()):
        storage_key = "cond_frame_outputs"
    else:
        storage_key = "non_cond_frame_outputs"
    current_out = output_dict.get(storage_key, {}).get(frame_idx)
    if current_out is None or "iou_score" not in current_out:
        return None
    score = obj_scores
    if isinstance(score, torch.Tensor):
        score_t = score.flatten()[obj_idx]
    else:
        score_t = torch.tensor(float(score[obj_idx]) if hasattr(score, "__getitem__") else float(score))
    object_score_norm = (score_t.sigmoid() * 2 - 1).clamp(min=0)
    iou_score = current_out["iou_score"].float().flatten()[obj_idx]
    return float((object_score_norm * iou_score).item())


def mask_quality_signals(raw_logits: np.ndarray, bin_mask: np.ndarray) -> Dict[str, float]:
    """energy_separation (mean logit inside minus outside the mask) and
    boundary_entropy (mean Bernoulli entropy in a 5 px band around the contour)."""
    out = {"energy_separation": NAN, "boundary_entropy": NAN}
    pos = bin_mask > 0
    neg = ~pos
    if pos.any():
        vals = raw_logits[pos]
        if neg.any():
            out["energy_separation"] = float(vals.mean() - raw_logits[neg].mean())
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil = cv2.dilate(bin_mask.astype(np.uint8), kern)
        ero = cv2.erode(bin_mask.astype(np.uint8), kern)
        band = (dil ^ ero) > 0
        if band.any():
            band_logits = np.clip(raw_logits[band].astype(np.float32), -30.0, 30.0)
            bp = 1.0 / (1.0 + np.exp(-band_logits))
            bp = np.clip(bp, 1e-7, 1.0 - 1e-7)
            out["boundary_entropy"] = float(
                -(bp * np.log(bp) + (1 - bp) * np.log(1 - bp)).mean()
            )
    return out


def apply_box_correction(tracker, inference_state, frame_idx: int, obj_id: int,
                         correction_bbox: List[int], video_w: int, video_h: int):
    """Add the corrector box as a SAM 3 box prompt at frame_idx and drop the
    cached output for that frame, so propagation recomputes it from the prompt."""
    x1, y1, x2, y2 = correction_bbox
    norm_box = torch.tensor(
        [x1 / video_w, y1 / video_h, x2 / video_w, y2 / video_h],
        dtype=torch.float32,
    )
    tracker.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=norm_box,
        rel_coordinates=True,
        clear_old_points=True,
        use_prev_mem_frame=False,
    )
    output_dict = inference_state["output_dict"]
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        output_dict[storage_key].pop(frame_idx, None)
    for key_set in inference_state.get("consolidated_frame_inds", {}).values():
        key_set.discard(frame_idx)


def apply_absence_correction(tracker, inference_state, frame_idx: int, obj_id: int,
                             video_w: int, video_h: int):
    """Add an empty-mask prompt at frame_idx (the corrector says not visible)."""
    empty_mask = torch.zeros((video_h, video_w), dtype=torch.float32)
    tracker.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=empty_mask,
    )
    output_dict = inference_state["output_dict"]
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        output_dict[storage_key].pop(frame_idx, None)
    for key_set in inference_state.get("consolidated_frame_inds", {}).values():
        key_set.discard(frame_idx)
