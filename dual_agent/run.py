#!/usr/bin/env python3
"""Run Dual-Agent (SAM 3 + VLM assessor + VLM corrector) on SeCVOS or LVOS-v2.

Writes to --output_dir:
  results.json   per object and frame: SAM 3 box, triggers, verdicts, corrections; plus a summary
  Annotations/   predicted masks in the dataset's own format (scripts/score_*.sh)
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import List

import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from agents import DualAgent
from benchmarks import DATASETS
from sam3_tracker import box_iou, build_tracker
from track import ObjectResult, ObjectTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_agents(args) -> DualAgent:
    from verifier import TrackingVerifier
    common = dict(seed_sampling=args.seed_sampling, resize=args.vlm_resize, max_pixels=args.max_pixels)
    assessor = TrackingVerifier(args.assessor_model_path, "assessment", args.assessor_format_prompt,
                                thinking=True, **common)
    corrector = TrackingVerifier(args.corrector_model_path, "correction", args.corrector_format_prompt,
                                 thinking=False, **common)
    return DualAgent(assessor, corrector, args)


def _finite(x):
    return None if x is None or math.isnan(x) or math.isinf(x) else float(x)


def to_json(result: ObjectResult) -> dict:
    frames = []
    for r in result.frames:
        frame = {
            "frame_idx": r.frame_idx,
            "sam3_bbox": r.sam3_bbox,
            "vlm_bbox": r.vlm_bbox,
            "gt_bbox": r.gt_bbox,
            "frame_annotated": r.frame_annotated,
            "objectness": _finite(r.objectness),
            "eff_iou": _finite(r.eff_iou),
            "is_occluded": r.is_occluded,
            "occluded_frames": r.occluded_frames,
            "frames_since_reappearance": r.frames_since_reappearance,
            "vlm_triggered": r.vlm_triggered,
            "vlm_corrected": r.vlm_corrected,
            "trigger_reason": r.trigger_reason,
        }
        if r.vlm_triggered:
            frame["assessor_verdict"] = r.assessor_verdict
        if r.vote_verdicts is not None:
            frame["vote_verdicts"] = r.vote_verdicts
            frame["corrector_called"] = r.corrector_called
        frames.append(frame)
    return {
        "sequence_id": result.sequence_id,
        "obj_id": result.obj_id,
        "assessor_calls": sum(r.vlm_triggered for r in result.frames),
        "corrector_calls": sum(r.corrector_called for r in result.frames),
        "corrections_applied": sum(r.vlm_corrected for r in result.frames),
        "frames": frames,
    }


def run_sequences(tracker, agents, dataset, seq_ids, args, masks_dir, position=0) -> List[dict]:
    objects = []
    for seq_id in tqdm(seq_ids, desc="Sequences", position=position):
        video = dataset.load(seq_id)
        if video is None:
            continue
        results = {}
        for obj_id in video.obj_ids:
            logger.info(f"{seq_id} obj {obj_id}")
            result = ObjectTracker(tracker, agents, video, obj_id, args).run()
            if result is not None:
                results[obj_id] = result
        if not results:
            continue
        masks = next(iter(results.values())).pred_masks
        if masks:
            height, width = next(iter(masks.values())).shape[:2]
        else:
            width, height = Image.open(video.frame_paths[0]).size
        dataset.save_masks(video, results, masks_dir, height, width)
        objects.extend(to_json(r) for r in results.values())
        logger.info(f"{seq_id}: {len(results)} objects, "
                    f"{sum(r.vlm_triggered for res in results.values() for r in res.frames)} assessor calls, "
                    f"{sum(r.vlm_corrected for res in results.values() for r in res.frames)} corrections")
        for result in results.values():
            result.pred_masks = {}
    return objects


def _worker(gpu_id, rank, seq_ids, args_dict, masks_dir, shard_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    agents = build_agents(args)
    objects = run_sequences(build_tracker(), agents, DATASETS[args.dataset](args.data_root), seq_ids, args,
                            masks_dir, position=rank)
    Path(shard_path).write_text(json.dumps(objects))


def run_sharded(seq_ids, args, masks_dir, results_path) -> List[dict]:
    """One process per GPU, sequences dealt round-robin."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [int(g) for g in visible.split(",")][:args.num_gpus] if visible else list(range(args.num_gpus))
    mp.set_start_method("spawn", force=True)
    shards, workers = [], []
    for rank, gpu_id in enumerate(gpu_ids):
        shard_path = f"{results_path}.shard{rank}.tmp"
        shards.append(shard_path)
        worker = mp.Process(target=_worker, args=(gpu_id, rank, seq_ids[rank::len(gpu_ids)], vars(args),
                                                  masks_dir, shard_path))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            logger.error(f"worker {worker.pid} exited with code {worker.exitcode}")
    objects = []
    for shard_path in shards:
        if os.path.exists(shard_path):
            objects.extend(json.loads(Path(shard_path).read_text()))
            os.remove(shard_path)
    return objects


def summarize(objects: List[dict]) -> dict:
    """Call counts, and where the frame is annotated, assessor accuracy (SAM 3 box
    IoU >= 0.5 counts as correct) and whether corrections moved the box closer to GT."""
    verdicts = dict(tp=0, tn=0, fp=0, fn=0, no_gt=0)
    corrections = dict(better=0, worse=0, neutral=0, no_gt=0)
    for obj in objects:
        for f in obj["frames"]:
            if not f["vlm_triggered"]:
                continue
            verdict = f.get("assessor_verdict")
            if verdict is not None:
                if f["gt_bbox"] is None or not f["frame_annotated"]:
                    verdicts["no_gt"] += 1
                else:
                    sam3_ok = (box_iou(f["sam3_bbox"], f["gt_bbox"]) if f["sam3_bbox"] else 0.0) >= 0.5
                    if verdict == "CORRECT":
                        verdicts["tn" if sam3_ok else "fn"] += 1
                    elif verdict == "INCORRECT":
                        verdicts["fp" if sam3_ok else "tp"] += 1
            if f["vlm_corrected"]:
                if f["gt_bbox"] is None:
                    corrections["no_gt"] += 1
                    continue
                before = box_iou(f["sam3_bbox"], f["gt_bbox"]) if f["sam3_bbox"] else 0.0
                after = box_iou(f["vlm_bbox"], f["gt_bbox"]) if f["vlm_bbox"] else 0.0
                key = "better" if after > before + 1e-6 else ("worse" if after < before - 1e-6 else "neutral")
                corrections[key] += 1
    return {
        "num_objects_evaluated": len(objects),
        "total_assessor_calls": sum(o["assessor_calls"] for o in objects),
        "total_corrector_calls": sum(o["corrector_calls"] for o in objects),
        "total_corrections": sum(o["corrections_applied"] for o in objects),
        **{f"assessor_{k}": v for k, v in verdicts.items()},
        **{f"corrections_{k}": v for k, v in corrections.items()},
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--data_root", required=True,
                   help="SeCVOS root, or the LVOS-v2 split directory (e.g. LVOSv2/valid)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--assessor_model_path", required=True, help="Qwen3-VL-8B-Thinking")
    p.add_argument("--corrector_model_path", required=True, help="corrector LoRA adapter directory")
    p.add_argument("--assessor_format_prompt", default="tracking_assessor_rich_format_prompt.txt")
    p.add_argument("--corrector_format_prompt", default="tracking_corrector_no_think_prompt.txt")
    p.add_argument("--video_ids_file", help="run only the video ids listed in this file")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--seed_sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_frame_details", action=argparse.BooleanOptionalAction, default=True,
                   help="write per-frame records to results.json")
    method = p.add_argument_group("method")
    method.add_argument("--streaming_trigger", action=argparse.BooleanOptionalAction, default=True,
                        help="effective-IoU variance trigger")
    method.add_argument("--auto_trigger_min_spacing", type=int, default=8)
    method.add_argument("--occlusion_poll_interval", type=int, default=12)
    method.add_argument("--reappearance_cooldown", type=int, default=8)
    method.add_argument("--num_first_appearance", type=int, default=2, help="GT exemplar frames")
    method.add_argument("--num_uniform", type=int, default=2, help="context frames spread over the older history")
    method.add_argument("--window_size", type=int, default=3, help="recent context frames + query")
    method.add_argument("--frame_stride", type=int, default=4, help="history sampling stride")
    method.add_argument("--vlm_resize", type=int, default=1000, help="long side of VLM input frames")
    method.add_argument("--max_pixels", type=int, default=401408)
    method.add_argument("--vote_count", type=int, default=3)
    method.add_argument("--vote_stride", type=int, default=4)
    method.add_argument("--max_correction_rounds", type=int, default=3)
    method.add_argument("--correction_iou_threshold", type=float, default=0.5)
    method.add_argument("--assessor_max_eff_iou", type=float, default=0.7,
                        help="no assessor call when SAM 3's effective IoU is above this")
    method.add_argument("--neg_prompt_max_eff_iou", type=float, default=0.7,
                        help="ignore 'not visible' when SAM 3's effective IoU is above this")
    return p.parse_args()


def main():
    args = parse_args()
    results_path = os.path.join(args.output_dir, "results.json")
    masks_dir = os.path.join(args.output_dir, "Annotations")
    os.makedirs(masks_dir, exist_ok=True)

    dataset = DATASETS[args.dataset](args.data_root)
    seq_ids = dataset.sequence_ids()
    if args.video_ids_file:
        keep = {line.strip() for line in Path(args.video_ids_file).read_text().splitlines() if line.strip()}
        seq_ids = [s for s in seq_ids if s in keep]
    args.num_gpus = max(1, min(args.num_gpus, len(seq_ids)))
    logger.info(f"{args.dataset}: {len(seq_ids)} sequences on {args.num_gpus} GPU(s) -> {args.output_dir}")

    if args.num_gpus > 1:
        objects = run_sharded(seq_ids, args, masks_dir, results_path)
    else:
        agents = build_agents(args)
        objects = run_sequences(build_tracker(), agents, dataset, seq_ids, args, masks_dir)

    summary = {"num_sequences": len(seq_ids), **summarize(objects), "args": vars(args)}
    if not args.save_frame_details:
        objects = [{k: v for k, v in o.items() if k != "frames"} for o in objects]
    Path(results_path).write_text(json.dumps({"summary": summary, "per_sequence": objects}, indent=2))
    logger.info(f"{summary['num_objects_evaluated']} objects, {summary['total_assessor_calls']} assessor calls, "
                f"{summary['total_corrector_calls']} corrector calls, {summary['total_corrections']} corrections")


if __name__ == "__main__":
    main()
