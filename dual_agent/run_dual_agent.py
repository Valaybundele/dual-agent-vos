#!/usr/bin/env python3
"""Dual-Agent evaluation on SeCVOS: SAM 3 tracker, VLM assessor and VLM corrector."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image as PILImage
from tqdm import tqdm

from sam3_io import (
    FrameRecord,
    SequenceResult,
    apply_absence_correction,
    apply_box_correction,
    build_tracker,
    eff_iou_score,
    get_gt_frame_indices,
    iou,
    load_annotations,
    load_frame_paths,
    make_mask_loader,
    mask_quality_signals,
    objectness,
    pred_to_bbox,
    save_masks,
)
from triggers import EffIouVarTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# SAM 3 reports the object present when its objectness logit is above this value.
OBJECTNESS_THRESHOLD = 0.0
# eff-IoU variance trigger parameters (window, std threshold, persistence).
EFF_IOU_VAR_W, EFF_IOU_VAR_THR, EFF_IOU_VAR_PERSIST = 4, 0.15, 2


def track_object(tracker, assessor, corrector, frame_paths: List[str], gt: Dict,
                 obj_id: int, seq_id: str, args, init_mask_fn) -> Optional[SequenceResult]:
    """Track one object through the video with the Dual-Agent loop."""

    init_frame = None
    for fidx in sorted(gt.keys()):
        if gt[fidx].get(obj_id) is not None:
            init_frame = fidx
            break
    if init_frame is None:
        logger.warning(f"  obj {obj_id}: no valid init frame, skipping")
        return None

    frame_dir = str(Path(frame_paths[0]).parent)
    try:
        inference_state = tracker.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
    except Exception as e:
        logger.error(f"  tracker.init_state failed: {e}")
        return None

    video_h = inference_state["video_height"]
    video_w = inference_state["video_width"]

    # Seed SAM 3 with the GT mask at the first annotated frame.
    seed_mask = init_mask_fn(init_frame, obj_id)
    if seed_mask is None:
        raise RuntimeError(f"{seq_id} obj {obj_id}: GT mask missing at init frame {init_frame}")
    if seed_mask.shape != (video_h, video_w):
        seed_mask = cv2.resize(seed_mask.astype(np.uint8), (video_w, video_h),
                               interpolation=cv2.INTER_NEAREST)
    tracker.add_new_mask(
        inference_state=inference_state,
        frame_idx=init_frame,
        obj_id=obj_id,
        mask=torch.from_numpy(seed_mask.astype(np.float32)),
    )

    num_frames = len(frame_paths)
    result = SequenceResult(sequence_id=seq_id, obj_id=obj_id)
    pred_masks: Dict[int, np.ndarray] = {}
    recorded_frames: set = set()

    # GT exemplars shown to both agents: the first annotated frames of the object.
    exemplar_paths: List[str] = []
    exemplar_bboxes: List[List[int]] = []
    exemplar_fidxs: List[int] = []
    max_exemplars = max(args.num_first_appearance, 2)
    for fidx in sorted(gt.keys()):
        gt_box = gt[fidx].get(obj_id)
        if gt_box is not None and fidx < num_frames:
            exemplar_paths.append(frame_paths[fidx])
            exemplar_bboxes.append(gt_box)
            exemplar_fidxs.append(fidx)
            if len(exemplar_paths) >= max_exemplars:
                break

    # Context buffer: (frame_idx, frame_path, sam3_bbox) every frame_stride frames.
    buffer: List[Tuple[int, str, List[int]]] = []
    frames_since_reappearance = 10**9
    was_occluded = False
    occluded_frames = 0
    prev_bin_mask: Optional[np.ndarray] = None
    streaming_trig = EffIouVarTrigger(w=EFF_IOU_VAR_W, thr=EFF_IOU_VAR_THR,
                                      persist=EFF_IOU_VAR_PERSIST,
                                      min_spacing=args.auto_trigger_min_spacing)
    # Last buffered frame before SAM 3 lost the object (kept in the recent context).
    trajectory_anchor: Optional[Tuple[int, str, List[int]]] = None
    vote_state = None

    start_frame = 0
    while start_frame < num_frames:
        correction_pending: Optional[Tuple[int, Optional[List[int]]]] = None

        for (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores) in \
                tracker.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_frame,
                    max_frame_num_to_track=num_frames,
                    reverse=False,
                    propagate_preflight=True,
                ):

            obj_id_list = [int(x) for x in obj_ids]
            if obj_id not in obj_id_list:
                start_frame = frame_idx + 1
                continue
            oi = obj_id_list.index(obj_id)

            logit = objectness(obj_scores, oi)
            eff_iou = eff_iou_score(inference_state, obj_scores, frame_idx, oi)
            sam3_bbox = pred_to_bbox(video_res_masks, oi, video_h, video_w) if logit > OBJECTNESS_THRESHOLD else None
            gt_bbox = gt.get(frame_idx, {}).get(obj_id)

            mask_logit_mean = float("nan")
            mask_logit_margin = float("nan")
            mask_logit_global_max = float("nan")
            n_components = -1
            pred_mask_area = 0
            boundary_entropy = float("nan")
            energy_separation = float("nan")
            temporal_mask_iou = float("nan")
            bin_m: Optional[np.ndarray] = None

            if video_res_masks is not None:
                raw_m = (video_res_masks[oi, 0] if video_res_masks.ndim == 4
                         else video_res_masks[oi]).cpu().numpy()
                if raw_m.shape != (video_h, video_w):
                    raw_m = cv2.resize(raw_m.astype(np.float32), (video_w, video_h),
                                       interpolation=cv2.INTER_LINEAR)
                bin_m = (raw_m > 0).astype(np.uint8)
                pred_masks[frame_idx] = bin_m
                pred_mask_area = int(bin_m.sum())
                try:
                    mask_logit_global_max = float(raw_m.max())
                except Exception:
                    mask_logit_global_max = float("nan")
                if pred_mask_area > 0:
                    pos = raw_m[bin_m > 0]
                    mask_logit_mean = float(pos.mean())
                    mask_logit_margin = float(pos.min())
                    try:
                        n_components = int(cv2.connectedComponents(bin_m, connectivity=8)[0]) - 1
                    except Exception:
                        n_components = -1
                    try:
                        sigs = mask_quality_signals(raw_m, bin_m)
                        be_v = sigs.get("boundary_entropy", float("nan"))
                        if be_v is not None:
                            boundary_entropy = float(be_v)
                        es_v = sigs.get("energy_separation", float("nan"))
                        if es_v is not None:
                            energy_separation = float(es_v)
                    except Exception:
                        boundary_entropy = float("nan")
                        energy_separation = float("nan")
                if (pred_mask_area > 0
                        and prev_bin_mask is not None
                        and prev_bin_mask.shape == bin_m.shape):
                    inter = int(np.logical_and(bin_m, prev_bin_mask).sum())
                    union = int(np.logical_or(bin_m, prev_bin_mask).sum())
                    if union > 0:
                        temporal_mask_iou = float(inter) / float(union)
            prev_bin_mask = bin_m if (bin_m is not None and bin_m.sum() > 0) else None

            rec = FrameRecord(
                frame_idx=frame_idx,
                sam3_bbox=sam3_bbox,
                vlm_bbox=sam3_bbox,
                gt_bbox=gt_bbox,
                frame_annotated=(obj_id in gt.get(frame_idx, {})),
                objectness=logit,
            )
            rec.eff_iou = float(eff_iou) if eff_iou is not None else float("nan")
            rec.mask_logit_mean = mask_logit_mean
            rec.mask_logit_margin = mask_logit_margin
            rec.mask_logit_global_max = mask_logit_global_max
            rec.n_components = n_components
            rec.boundary_entropy = boundary_entropy
            rec.energy_separation = energy_separation
            rec.pred_mask_area = pred_mask_area
            rec.temporal_mask_iou = temporal_mask_iou

            # After a correction, propagation restarts at the corrected frame;
            # frames already recorded are not processed again.
            if frame_idx in recorded_frames:
                start_frame = frame_idx + 1
                continue

            if frame_idx % args.frame_stride == 0:
                # Exemplar frames are not repeated as context.
                if sam3_bbox is not None and frame_idx not in exemplar_fidxs:
                    buffer.append((frame_idx, frame_paths[frame_idx], sam3_bbox))

            # Occlusion state
            is_occluded = logit <= OBJECTNESS_THRESHOLD
            if not was_occluded and is_occluded and sam3_bbox is None:
                if buffer:
                    trajectory_anchor = buffer[-1]
            if is_occluded:
                occluded_frames += 1
            else:
                occluded_frames = 0
                trajectory_anchor = None
            if was_occluded and not is_occluded:
                frames_since_reappearance = 0
            else:
                frames_since_reappearance += 1

            rec.is_occluded = bool(is_occluded)
            rec.occluded_frames = int(occluded_frames)
            rec.frames_since_reappearance = int(frames_since_reappearance)

            # Vote frame of a pending INCORRECT verdict
            if vote_state is not None and frame_idx in vote_state["check_frames"]:
                vs = vote_state
                v_query_pil = PILImage.open(frame_paths[frame_idx]).convert("RGB")
                v_fidxs = vs["base_fidxs"][:-1] + [frame_idx]

                assessor.set_sampling_identity(seq_id, obj_id, frame_idx, "assess")
                vr = assessor.verify(
                    context_frames=vs["ctx_frames"],
                    context_bboxes=vs["ctx_bboxes"],
                    query_frame=v_query_pil,
                    query_bbox=sam3_bbox,
                    video_frame_indices=v_fidxs,
                    num_uniform_override=vs["num_uniform"],
                    num_first_appearance_override=vs["nfa_override"],
                )
                vs["verdicts"].append(vr.assessment)
                vs["check_frames"].discard(frame_idx)

                n_votes = len(vs["verdicts"])
                n_incorrect = sum(1 for v in vs["verdicts"] if v == "INCORRECT")
                n_remaining = len(vs["check_frames"])
                total_possible = n_votes + n_remaining
                logger.debug(f"  f{frame_idx}: vote {n_votes}/{total_possible} "
                             f"-> {vr.assessment} ({n_incorrect} INCORRECT)")

                if n_incorrect > total_possible // 2:
                    # Majority INCORRECT: run the corrector on this frame.
                    rec.vlm_triggered = True
                    rec.assessor_verdict = "INCORRECT"
                    rec.trigger_reason = vs["trigger_reason"]
                    rec.corrector_called = True
                    rec.corrector_future_fidxs = []
                    rec.vote_verdicts = list(vs["verdicts"])

                    correction_bbox = None
                    correction_visible = None
                    current_query_bbox = sam3_bbox
                    for corr_round in range(args.max_correction_rounds):
                        corrector.set_sampling_identity(seq_id, obj_id, frame_idx, f"corr{corr_round}")
                        cr = corrector.verify(
                            context_frames=vs["ctx_frames"],
                            context_bboxes=vs["ctx_bboxes"],
                            query_frame=v_query_pil,
                            query_bbox=current_query_bbox,
                            video_frame_indices=v_fidxs,
                            num_uniform_override=vs["num_uniform"],
                            num_first_appearance_override=vs["nfa_override"],
                        )
                        correction_bbox = cr.predicted_bbox
                        correction_visible = cr.visible
                        if correction_bbox is None:
                            break
                        if corr_round < args.max_correction_rounds - 1:
                            # The assessor re-checks the proposed box; retry if rejected.
                            assessor.set_sampling_identity(seq_id, obj_id, frame_idx, f"recheck{corr_round}")
                            recheck = assessor.verify(
                                context_frames=vs["ctx_frames"],
                                context_bboxes=vs["ctx_bboxes"],
                                query_frame=v_query_pil,
                                query_bbox=correction_bbox,
                                video_frame_indices=v_fidxs,
                                num_uniform_override=vs["num_uniform"],
                                num_first_appearance_override=vs["nfa_override"],
                            )
                            if recheck.assessment == "CORRECT":
                                break
                            current_query_bbox = correction_bbox

                    if correction_bbox is not None:
                        corr_iou = iou(correction_bbox, sam3_bbox) if sam3_bbox else 0.0
                        if corr_iou < args.correction_iou_threshold:
                            rec.vlm_bbox = correction_bbox
                            rec.vlm_corrected = True
                            correction_pending = (frame_idx, correction_bbox)
                    elif correction_visible is False and sam3_bbox is not None:
                        if eff_iou is not None and eff_iou > args.neg_prompt_max_eff_iou:
                            logger.debug(f"  f{frame_idx}: SAM3 eff_iou={eff_iou:.3f} > "
                                         f"{args.neg_prompt_max_eff_iou} -- ignoring corrector visible=False")
                        else:
                            rec.vlm_bbox = None
                            rec.vlm_corrected = True
                            correction_pending = (frame_idx, None)

                    vote_state = None

                elif n_incorrect + n_remaining <= total_possible // 2:
                    logger.debug(f"  f{frame_idx}: vote abandoned, majority CORRECT")
                    rec.vote_verdicts = list(vs["verdicts"])
                    vote_state = None
                elif n_votes >= args.vote_count:
                    logger.debug(f"  f{frame_idx}: vote done, no majority INCORRECT")
                    rec.vote_verdicts = list(vs["verdicts"])
                    vote_state = None

            # Trigger
            elif vote_state is None:
                trigger_vlm = False
                trigger_reason = None
                prev_last_fire_t = None
                if not is_occluded and frames_since_reappearance == args.reappearance_cooldown:
                    trigger_vlm = True
                    trigger_reason = "reappear"
                elif is_occluded and occluded_frames % args.occlusion_poll_interval == 0:
                    trigger_vlm = True
                    trigger_reason = "occ_poll"
                else:
                    prev_last_fire_t = streaming_trig.last_fire_t
                    if streaming_trig.step(eff_iou if eff_iou is not None else float("nan")):
                        trigger_vlm = True
                        trigger_reason = "streaming_var"

                # SAM 3 is confident: skip the call (and give back the trigger's spacing).
                if (trigger_vlm
                        and eff_iou is not None
                        and eff_iou > args.assessor_max_eff_iou):
                    trigger_vlm = False
                    logger.debug(f"  f{frame_idx}: eff_iou_short_circuit "
                                 f"(eff={eff_iou:.3f} > {args.assessor_max_eff_iou}), skipping assessor")
                    if trigger_reason == "streaming_var":
                        streaming_trig.last_fire_t = prev_last_fire_t

                if trigger_vlm and len(exemplar_paths) > 0 and len(buffer) >= 2:
                    query_idx = frame_idx
                    query_path = frame_paths[frame_idx]
                    query_sam3 = sam3_bbox

                    # Context: exemplars + uniform earlier frames + recent frames
                    if trigger_reason == "occ_poll":
                        n_ex = min(2, len(exemplar_paths))
                        n_recent = 2
                        n_uni = 2
                    else:
                        n_ex = min(args.num_first_appearance, len(exemplar_paths))
                        n_recent = args.window_size - 1
                        n_uni = args.num_uniform

                    ex_p = exemplar_paths[:n_ex]
                    ex_b = exemplar_bboxes[:n_ex]
                    ex_f = exemplar_fidxs[:n_ex]

                    ctx_buf = [e for e in buffer if e[0] < query_idx]
                    if trajectory_anchor is not None and trajectory_anchor[0] < query_idx:
                        anchor_idx = trajectory_anchor[0]
                        ctx_no_anchor = [e for e in ctx_buf if e[0] != anchor_idx]
                        recent_entries = ctx_no_anchor[-(n_recent - 1):] + [trajectory_anchor]
                        recent_entries.sort(key=lambda e: e[0])
                    else:
                        recent_entries = ctx_buf[-n_recent:]

                    window_start_idx = recent_entries[0][0] if recent_entries else query_idx
                    earlier = [e for e in buffer if e[0] < window_start_idx]
                    if len(earlier) >= n_uni:
                        step = len(earlier) / n_uni
                        uniform_entries = [earlier[int(i * step)] for i in range(n_uni)]
                    elif earlier:
                        uniform_entries = list(earlier)
                    else:
                        uniform_entries = []

                    ex_frames = [PILImage.open(p).convert("RGB") for p in ex_p]
                    uni_frames = [PILImage.open(p).convert("RGB") for _, p, _ in uniform_entries]
                    ctx_frames = [PILImage.open(p).convert("RGB") for _, p, _ in recent_entries]
                    query_pil = PILImage.open(query_path).convert("RGB")

                    all_ctx_frames = ex_frames + uni_frames + ctx_frames
                    all_ctx_bboxes = (list(ex_b)
                                      + [b for _, _, b in uniform_entries]
                                      + [b for _, _, b in recent_entries])
                    all_fidxs = (list(ex_f)
                                 + [fi for fi, _, _ in uniform_entries]
                                 + [fi for fi, _, _ in recent_entries]
                                 + [query_idx])
                    nfa_override = n_ex if n_ex != args.num_first_appearance else None

                    rec.vlm_triggered = True
                    corrector_called = False

                    # Assessor
                    assessor.set_sampling_identity(seq_id, obj_id, query_idx, "assess")
                    ar = assessor.verify(
                        context_frames=all_ctx_frames,
                        context_bboxes=all_ctx_bboxes,
                        query_frame=query_pil,
                        query_bbox=query_sam3,
                        video_frame_indices=all_fidxs,
                        num_uniform_override=len(uniform_entries),
                        num_first_appearance_override=nfa_override,
                    )
                    assessor_verdict = ar.assessment
                    needs_correction = (assessor_verdict == "INCORRECT")

                    # INCORRECT: confirm by majority vote on later frames
                    if needs_correction and args.vote_count > 1:
                        check_frames = set()
                        for vi in range(1, args.vote_count):
                            vf = query_idx + args.vote_stride * vi
                            if vf < num_frames:
                                check_frames.add(vf)
                        if check_frames:
                            vote_state = {
                                "check_frames": check_frames,
                                "verdicts": ["INCORRECT"],
                                "ctx_frames": all_ctx_frames,
                                "ctx_bboxes": all_ctx_bboxes,
                                "base_fidxs": all_fidxs,
                                "nfa_override": nfa_override,
                                "num_uniform": len(uniform_entries),
                                "trigger_reason": trigger_reason,
                            }
                            needs_correction = False
                            logger.debug(f"  f{query_idx}: INCORRECT, voting at {sorted(check_frames)}")

                    # No vote frames left (end of video): correct now
                    correction_bbox = None
                    correction_visible = None
                    if needs_correction:
                        corrector_called = True
                        rec.corrector_future_fidxs = []
                        current_query_bbox = query_sam3
                        for corr_round in range(args.max_correction_rounds):
                            corrector.set_sampling_identity(seq_id, obj_id, query_idx, f"corr{corr_round}")
                            cr = corrector.verify(
                                context_frames=all_ctx_frames,
                                context_bboxes=all_ctx_bboxes,
                                query_frame=query_pil,
                                query_bbox=current_query_bbox,
                                video_frame_indices=all_fidxs,
                                num_uniform_override=len(uniform_entries),
                                num_first_appearance_override=nfa_override,
                            )
                            correction_bbox = cr.predicted_bbox
                            correction_visible = cr.visible
                            if correction_bbox is None:
                                break
                            if corr_round < args.max_correction_rounds - 1:
                                assessor.set_sampling_identity(seq_id, obj_id, query_idx, f"recheck{corr_round}")
                                recheck = assessor.verify(
                                    context_frames=all_ctx_frames,
                                    context_bboxes=all_ctx_bboxes,
                                    query_frame=query_pil,
                                    query_bbox=correction_bbox,
                                    video_frame_indices=all_fidxs,
                                    num_uniform_override=len(uniform_entries),
                                    num_first_appearance_override=nfa_override,
                                )
                                if recheck.assessment == "CORRECT":
                                    break
                                current_query_bbox = correction_bbox

                        if correction_bbox is not None:
                            corr_iou = iou(correction_bbox, query_sam3) if query_sam3 else 0.0
                            if corr_iou < args.correction_iou_threshold:
                                rec.vlm_bbox = correction_bbox
                                rec.vlm_corrected = True
                                correction_pending = (query_idx, correction_bbox)
                        elif correction_visible is False and query_sam3 is not None:
                            if eff_iou is not None and eff_iou > args.neg_prompt_max_eff_iou:
                                logger.debug(f"  f{query_idx}: SAM3 eff_iou={eff_iou:.3f} > "
                                             f"{args.neg_prompt_max_eff_iou} -- ignoring corrector visible=False")
                            else:
                                rec.vlm_bbox = None
                                rec.vlm_corrected = True
                                correction_pending = (query_idx, None)

                    rec.assessor_verdict = assessor_verdict
                    rec.trigger_reason = trigger_reason
                    rec.corrector_called = corrector_called
                    rec.vote_verdicts = [assessor_verdict]

            was_occluded = is_occluded
            result.frames.append(rec)
            recorded_frames.add(frame_idx)
            start_frame = frame_idx + 1

            if correction_pending is not None:
                break

        if correction_pending is not None:
            # Give the correction to SAM 3 and resume propagation from that frame.
            corr_frame, corr_bbox = correction_pending
            if corr_bbox is not None:
                apply_box_correction(tracker, inference_state, corr_frame, obj_id,
                                     corr_bbox, video_w, video_h)
            else:
                apply_absence_correction(tracker, inference_state, corr_frame, obj_id,
                                         video_w, video_h)
            buffer = [(fi, p, b) for fi, p, b in buffer if fi < corr_frame]
            was_occluded = (corr_bbox is None)
            occluded_frames = 0
            trajectory_anchor = None
            start_frame = corr_frame
        else:
            break

    result.pred_masks = pred_masks
    return result



def _safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def serialize_result(res: SequenceResult) -> dict:
    entry = {
        "sequence_id": res.sequence_id,
        "obj_id": res.obj_id,
        "assessor_calls": res.vlm_calls,
        "corrector_calls": sum(1 for fr in res.frames if fr.corrector_called),
        "corrections_applied": res.vlm_corrections,
        "frames": [],
    }
    for fr in res.frames:
        fd = {
            "frame_idx": fr.frame_idx,
            "sam3_bbox": fr.sam3_bbox,
            "vlm_bbox": fr.vlm_bbox,
            "gt_bbox": fr.gt_bbox,
            "frame_annotated": fr.frame_annotated,
            "objectness": _safe_float(fr.objectness),
            "vlm_triggered": fr.vlm_triggered,
            "vlm_corrected": fr.vlm_corrected,
            "trigger_reason": fr.trigger_reason or None,
            "eff_iou": _safe_float(fr.eff_iou),
            "mask_logit_mean": _safe_float(fr.mask_logit_mean),
            "mask_logit_margin": _safe_float(fr.mask_logit_margin),
            "mask_logit_global_max": _safe_float(fr.mask_logit_global_max),
            "n_components": int(fr.n_components),
            "boundary_entropy": _safe_float(fr.boundary_entropy),
            "energy_separation": _safe_float(fr.energy_separation),
            "temporal_mask_iou": _safe_float(fr.temporal_mask_iou),
            "pred_mask_area": int(fr.pred_mask_area),
            "is_occluded": bool(fr.is_occluded),
            "occluded_frames": int(fr.occluded_frames),
            "frames_since_reappearance": int(fr.frames_since_reappearance),
        }
        if fr.vlm_triggered:
            fd["assessor_verdict"] = fr.assessor_verdict
        if fr.vote_verdicts is not None:
            fd["vote_verdicts"] = list(fr.vote_verdicts)
            fd["vote_logprobs"] = [None] * len(fr.vote_verdicts)
            fd["corrector_called"] = fr.corrector_called
        if fr.corrector_future_fidxs is not None:
            fd["corrector_future_fidxs"] = list(fr.corrector_future_fidxs)
        entry["frames"].append(fd)
    return entry



def run_sequences(tracker, assessor, corrector, seq_ids, args, masks_output_dir,
                  tqdm_position=0) -> List[dict]:
    masks_root = os.path.join(args.secvos_root, "Annotations")
    frames_root = os.path.join(args.secvos_root, "JPEGImages")
    all_results: List[SequenceResult] = []

    for seq_id in tqdm(seq_ids, desc="Sequences", position=tqdm_position):
        frame_paths = load_frame_paths(frames_root, seq_id)
        if not frame_paths:
            logger.warning(f"No frames for {seq_id}, skipping")
            continue
        gt = load_annotations(masks_root, seq_id)
        gt_fidxs = get_gt_frame_indices(masks_root, seq_id)
        init_mask_fn = make_mask_loader(masks_root, seq_id)
        if not gt:
            logger.warning(f"No annotations for {seq_id}, skipping")
            continue

        obj_ids = {oid for fgt in gt.values() for oid in fgt}
        seq_results: Dict[int, SequenceResult] = {}
        for obj_id in sorted(obj_ids):
            logger.info(f"  {seq_id} obj={obj_id}")
            res = track_object(tracker, assessor, corrector, frame_paths, gt, obj_id,
                               seq_id, args, init_mask_fn)
            if res is not None:
                all_results.append(res)
                seq_results[obj_id] = res

        if seq_results:
            first_res = next(iter(seq_results.values()))
            if first_res.pred_masks:
                vh, vw = next(iter(first_res.pred_masks.values())).shape[:2]
            else:
                vw, vh = PILImage.open(frame_paths[0]).size
            save_masks(seq_results, masks_output_dir, seq_id, vh, vw, gt_frame_indices=gt_fidxs)
            n_calls = sum(r.vlm_calls for r in seq_results.values())
            n_corr = sum(r.vlm_corrections for r in seq_results.values())
            logger.info(f"  [{seq_id}] objs={len(seq_results)} assessor_calls={n_calls} corrections={n_corr}")
            for res in seq_results.values():
                res.pred_masks = {}

    return [serialize_result(r) for r in all_results]


def build_agents(args):
    """Load the assessor (8B, thinking, sampled) and the corrector (4B + LoRA, no-think, greedy)."""
    from verifier import TrackingVerifier
    logger.info(f"Loading assessor model (think=True, resize={args.vlm_resize}, max_pixels={args.max_pixels})...")
    assessor = TrackingVerifier(
        model_path=args.assessor_model_path,
        task_schema="assessment",
        format_prompt=args.assessor_format_prompt,
        num_first_appearance=args.num_first_appearance,
        num_uniform=args.num_uniform,
        nothink=False,
        seed_sampling=args.seed_sampling,
        resize_size=args.vlm_resize,
        max_pixels=args.max_pixels,
    )
    logger.info(f"Loading corrector model (think=False, resize={args.vlm_resize}, max_pixels={args.max_pixels})...")
    corrector = TrackingVerifier(
        model_path=args.corrector_model_path,
        task_schema="correction",
        format_prompt=args.corrector_format_prompt,
        num_first_appearance=args.num_first_appearance,
        num_uniform=args.num_uniform,
        nothink=True,
        seed_sampling=args.seed_sampling,
        resize_size=args.vlm_resize,
        max_pixels=args.max_pixels,
    )
    return assessor, corrector


def _gpu_worker(gpu_id, rank, seq_ids, args_dict, masks_output_dir, shard_path):
    """One process per GPU (--num_gpus > 1): load the models and run a shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    assessor, corrector = build_agents(args)
    tracker = build_tracker()
    per_seq_data = run_sequences(tracker, assessor, corrector, seq_ids, args,
                                 masks_output_dir, tqdm_position=rank)
    with open(shard_path, "w") as f:
        json.dump(per_seq_data, f)
    print(f"GPU {gpu_id}: {len(per_seq_data)} objects from {len(seq_ids)} sequences -> {shard_path}")


def summarize(per_seq_data: List[dict]) -> dict:
    """Call counts and, where GT is annotated, assessor/correction outcomes (box IoU)."""
    total_calls = sum(e["assessor_calls"] for e in per_seq_data)
    total_corr = sum(e["corrections_applied"] for e in per_seq_data)
    total_corrector_calls = sum(e["corrector_calls"] for e in per_seq_data)

    tp = tn = fp = fn = no_gt = 0
    corr = {"better": 0, "worse": 0, "neutral": 0, "no_gt": 0}
    for e in per_seq_data:
        for fr in e["frames"]:
            if not fr["vlm_triggered"]:
                continue
            verdict = fr.get("assessor_verdict")
            if verdict is not None:
                if fr["gt_bbox"] is None or not fr["frame_annotated"]:
                    no_gt += 1
                else:
                    sam3_ok = (iou(fr["sam3_bbox"], fr["gt_bbox"]) if fr["sam3_bbox"] else 0.0) >= 0.5
                    if verdict == "CORRECT":
                        tn, fn = (tn + 1, fn) if sam3_ok else (tn, fn + 1)
                    elif verdict == "INCORRECT":
                        fp, tp = (fp + 1, tp) if sam3_ok else (fp, tp + 1)
            if fr["vlm_corrected"]:
                if fr["gt_bbox"] is None:
                    corr["no_gt"] += 1
                    continue
                s_iou = iou(fr["sam3_bbox"], fr["gt_bbox"]) if fr["sam3_bbox"] else 0.0
                v_iou = iou(fr["vlm_bbox"], fr["gt_bbox"]) if fr["vlm_bbox"] else 0.0
                key = "better" if v_iou > s_iou + 1e-6 else ("worse" if v_iou < s_iou - 1e-6 else "neutral")
                corr[key] += 1

    return {
        "num_objects_evaluated": len(per_seq_data),
        "total_assessor_calls": total_calls,
        "total_corrector_calls": total_corrector_calls,
        "total_corrections": total_corr,
        "assessor_tp": tp, "assessor_tn": tn, "assessor_fp": fp, "assessor_fn": fn,
        "assessor_no_gt": no_gt,
        "corrections_better": corr["better"],
        "corrections_worse": corr["worse"],
        "corrections_neutral": corr["neutral"],
        "corrections_no_gt": corr["no_gt"],
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-Agent (SAM 3 + VLM assessor + VLM corrector) on SeCVOS")
    parser.add_argument("--secvos_root", required=True,
                        help="SeCVOS root with JPEGImages/, Annotations/ (and video_id.txt)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--assessor_model_path", required=True, help="Qwen3-VL-8B-Thinking")
    parser.add_argument("--corrector_model_path", required=True,
                        help="corrector LoRA adapter directory (checkpoint-9418)")
    parser.add_argument("--assessor_format_prompt", default="tracking_assessor_rich_format_prompt.txt")
    parser.add_argument("--corrector_format_prompt", default="tracking_corrector_no_think_prompt.txt")
    parser.add_argument("--video_ids_file", default=None, help="optional: one video id per line")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs; sequences are sharded across them")
    parser.add_argument("--seed_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_frame_details", action=argparse.BooleanOptionalAction, default=True,
                        help="write per-frame records to results.json")
    # Method parameters; defaults are the paper configuration.
    parser.add_argument("--occlusion_poll_interval", type=int, default=12)
    parser.add_argument("--reappearance_cooldown", type=int, default=8)
    parser.add_argument("--auto_trigger_min_spacing", type=int, default=8)
    parser.add_argument("--num_first_appearance", type=int, default=2, help="GT exemplar frames")
    parser.add_argument("--num_uniform", type=int, default=2, help="uniformly sampled earlier context frames")
    parser.add_argument("--window_size", type=int, default=3, help="recent context frames + query")
    parser.add_argument("--frame_stride", type=int, default=4, help="context buffer stride")
    parser.add_argument("--vlm_resize", type=int, default=1000, help="long side of VLM input frames")
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--vote_count", type=int, default=3)
    parser.add_argument("--vote_stride", type=int, default=4)
    parser.add_argument("--max_correction_rounds", type=int, default=3)
    parser.add_argument("--correction_iou_threshold", type=float, default=0.5)
    parser.add_argument("--assessor_max_eff_iou", type=float, default=0.7,
                        help="skip the assessor when SAM 3 eff-IoU is above this")
    parser.add_argument("--neg_prompt_max_eff_iou", type=float, default=0.7,
                        help="ignore 'not visible' when SAM 3 eff-IoU is above this")
    args = parser.parse_args()

    output_json = os.path.join(args.output_dir, "results.json")
    masks_output_dir = os.path.join(args.output_dir, "Annotations")
    os.makedirs(masks_output_dir, exist_ok=True)

    # Sequences: video_id.txt if present, else the annotation folders
    id_file = os.path.join(args.secvos_root, "video_id.txt")
    if os.path.exists(id_file):
        with open(id_file) as f:
            seq_ids = sorted(ln.strip() for ln in f if ln.strip())
    else:
        seq_ids = [e.name for e in sorted(Path(args.secvos_root, "Annotations").iterdir()) if e.is_dir()]
    if args.video_ids_file:
        with open(args.video_ids_file) as f:
            keep = {ln.strip() for ln in f if ln.strip()}
        seq_ids = [s for s in seq_ids if s in keep]
        logger.info(f"Filtered to {len(seq_ids)} sequences from --video_ids_file")

    num_gpus = args.num_gpus
    num_gpus = max(1, min(num_gpus, len(seq_ids)))
    logger.info(f"SeCVOS | {len(seq_ids)} sequences | gpus={num_gpus} | masks -> {masks_output_dir}")

    if num_gpus > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_ids = ([int(g) for g in visible.split(",")][:num_gpus] if visible
                   else list(range(num_gpus)))
        shards = [[] for _ in gpu_ids]
        for i, sid in enumerate(seq_ids):
            shards[i % len(gpu_ids)].append(sid)
        mp.set_start_method("spawn", force=True)
        shard_paths, processes = [], []
        for rank, gpu_id in enumerate(gpu_ids):
            sp = output_json + f".shard{rank}.tmp"
            shard_paths.append(sp)
            p = mp.Process(target=_gpu_worker,
                           args=(gpu_id, rank, shards[rank], vars(args), masks_output_dir, sp))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker PID {p.pid} exited with code {p.exitcode}")
        per_seq_data = []
        for sp in shard_paths:
            if Path(sp).exists():
                per_seq_data.extend(json.loads(Path(sp).read_text()))
                os.remove(sp)
    else:
        assessor, corrector = build_agents(args)
        tracker = build_tracker()
        per_seq_data = run_sequences(tracker, assessor, corrector, seq_ids, args, masks_output_dir)

    summary = {"num_sequences": len(seq_ids), "num_gpus": num_gpus}
    summary.update(summarize(per_seq_data))
    summary.update({"masks_dir": masks_output_dir, "args": vars(args)})

    per_sequence = per_seq_data
    if not args.save_frame_details:
        per_sequence = [{k: v for k, v in e.items() if k != "frames"} for e in per_seq_data]
    with open(output_json, "w") as f:
        json.dump({"summary": summary, "per_sequence": per_sequence}, f, indent=2)

    logger.info(f"Objects: {summary['num_objects_evaluated']} | assessor calls: "
                f"{summary['total_assessor_calls']} | corrector calls: "
                f"{summary['total_corrector_calls']} | corrections: {summary['total_corrections']}")
    logger.info(f"Saved -> {output_json}")
    logger.info(f"Masks -> {masks_output_dir}  (score with scripts/score_secvos.sh)")


if __name__ == "__main__":
    main()
