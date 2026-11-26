"""The Dual-Agent loop for one object.

SAM 3 is seeded with the object's GT mask at its first annotated frame and
propagates through the video. A trigger fires when
  - reappear:      SAM 3 re-detected the object `reappearance_cooldown` frames ago,
  - occ_poll:      every `occlusion_poll_interval` frames while SAM 3 reports it absent,
  - streaming_var: SAM 3's effective IoU becomes unstable (EffIouVarTrigger, optional).
Unless SAM 3 is confident (effective IoU > assessor_max_eff_iou), the assessor
judges SAM 3's box. An INCORRECT verdict is confirmed by a majority vote over
`vote_count` frames `vote_stride` apart; then the corrector proposes a box or
says the object is not visible, and that answer is given to SAM 3 as a prompt.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import torch

from agents import Context, Correction, DualAgent, Sample, build_context, open_rgb
from sam3_tracker import effective_iou, mask_logits, mask_to_box, objectness, prompt_absent, prompt_box
from triggers import EffIouVarTrigger

logger = logging.getLogger(__name__)

# SAM 3 reports the object present when its objectness logit is above this value.
OBJECTNESS_THRESHOLD = 0.0
EFF_IOU_VAR_WINDOW, EFF_IOU_VAR_THRESHOLD, EFF_IOU_VAR_PERSIST = 4, 0.15, 2


@dataclass
class FrameRecord:
    frame_idx: int
    sam3_bbox: Optional[List[int]]   # None when SAM 3 reports the object absent
    vlm_bbox: Optional[List[int]]    # box after the Dual-Agent step (sam3_bbox unless corrected)
    gt_bbox: Optional[List[int]]
    frame_annotated: bool
    objectness: float
    eff_iou: float
    is_occluded: bool = False
    occluded_frames: int = 0
    frames_since_reappearance: int = 0
    vlm_triggered: bool = False      # the assessor judged this frame as a trigger or a deciding vote
    vlm_corrected: bool = False      # a correction was given to SAM 3 at this frame
    trigger_reason: Optional[str] = None
    assessor_verdict: Optional[str] = None
    vote_verdicts: Optional[List[Optional[str]]] = None
    corrector_called: bool = False


@dataclass
class ObjectResult:
    sequence_id: str
    obj_id: int
    frames: List[FrameRecord] = field(default_factory=list)
    pred_masks: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class PendingVote:
    frames: Set[int]
    verdicts: List[Optional[str]]
    context: Context
    trigger_reason: str


class ObjectTracker:
    def __init__(self, tracker, agents: DualAgent, video, obj_id: int, args):
        self.tracker = tracker
        self.agents = agents
        self.video = video
        self.obj_id = obj_id
        self.args = args
        self.num_frames = len(video.frame_paths)

    def run(self) -> Optional[ObjectResult]:
        video, obj_id = self.video, self.obj_id
        seed_frame = next((f for f in sorted(video.gt_boxes) if video.gt_boxes[f].get(obj_id) is not None), None)
        if seed_frame is None:
            logger.warning(f"{video.seq_id} obj {obj_id}: no annotated frame, skipping")
            return None
        try:
            self.state = self.tracker.init_state(video_path=str(Path(video.frame_paths[0]).parent),
                                                 offload_video_to_cpu=False, offload_state_to_cpu=False)
        except Exception as e:
            logger.error(f"tracker.init_state failed: {e}")
            return None
        self.height, self.width = self.state["video_height"], self.state["video_width"]
        self._seed(seed_frame)

        self.result = ObjectResult(video.seq_id, obj_id)
        self.recorded: Set[int] = set()
        self.exemplars = self._exemplars()
        self.exemplar_frames = {s.frame_idx for s in self.exemplars}
        self.history: List[Sample] = []
        self.anchor: Optional[Sample] = None
        self.was_occluded = False
        self.occluded_frames = 0
        self.frames_since_reappearance = 10**9
        self.vote: Optional[PendingVote] = None
        self.streaming = (EffIouVarTrigger(EFF_IOU_VAR_WINDOW, EFF_IOU_VAR_THRESHOLD, EFF_IOU_VAR_PERSIST,
                                           self.args.auto_trigger_min_spacing)
                          if self.args.streaming_trigger else None)

        start = 0
        while start < self.num_frames:
            correction, start = self._propagate(start)
            if correction is None:
                break
            self._commit(correction)
            # Restart at the corrected frame so SAM 3 re-predicts it from the prompt.
            start = correction.frame_idx
        return self.result

    def _seed(self, frame_idx: int):
        mask = self.video.gt_mask(frame_idx, self.obj_id)
        if mask is None:
            raise RuntimeError(f"{self.video.seq_id} obj {self.obj_id}: no GT mask at seed frame {frame_idx}")
        if mask.shape != (self.height, self.width):
            mask = cv2.resize(mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.tracker.add_new_mask(inference_state=self.state, frame_idx=frame_idx, obj_id=self.obj_id,
                                  mask=torch.from_numpy(mask.astype(np.float32)))

    def _exemplars(self) -> List[Sample]:
        exemplars = []
        for frame_idx in sorted(self.video.gt_boxes):
            box = self.video.gt_boxes[frame_idx].get(self.obj_id)
            if box is not None and frame_idx < self.num_frames:
                exemplars.append(Sample(frame_idx, self.video.frame_paths[frame_idx], box))
                if len(exemplars) >= max(self.args.num_first_appearance, 2):
                    break
        return exemplars

    def _propagate(self, start: int):
        """Propagate from `start` until a correction is needed. Returns (correction, next start)."""
        for frame_idx, obj_ids, _, video_res_masks, obj_scores in self.tracker.propagate_in_video(
                self.state, start_frame_idx=start, max_frame_num_to_track=self.num_frames,
                reverse=False, propagate_preflight=True):
            obj_ids = [int(x) for x in obj_ids]
            start = frame_idx + 1
            if self.obj_id not in obj_ids:
                continue
            index = obj_ids.index(self.obj_id)
            logit = objectness(obj_scores, index)
            eff_iou = effective_iou(self.state, obj_scores, frame_idx, index)
            sam3_box = None
            if video_res_masks is not None:
                logits = mask_logits(video_res_masks, index, self.height, self.width)
                self.result.pred_masks[frame_idx] = (logits > 0).astype(np.uint8)
                if logit > OBJECTNESS_THRESHOLD:
                    sam3_box = mask_to_box((logits > 0).astype(np.uint8))
            # After a correction the corrected frame is yielded again: keep its new
            # mask, but it has already been through the agents.
            if frame_idx in self.recorded:
                continue

            gt = self.video.gt_boxes.get(frame_idx, {})
            record = FrameRecord(frame_idx, sam3_box, sam3_box, gt.get(self.obj_id), self.obj_id in gt, logit,
                                 float(eff_iou) if eff_iou is not None else float("nan"))
            correction = self._step(record, sam3_box, eff_iou, logit <= OBJECTNESS_THRESHOLD)
            self.result.frames.append(record)
            self.recorded.add(frame_idx)
            if correction is not None:
                return correction, start
        return None, start

    def _step(self, record: FrameRecord, sam3_box, eff_iou, occluded: bool) -> Optional[Correction]:
        frame_idx = record.frame_idx
        if frame_idx % self.args.frame_stride == 0 and sam3_box is not None \
                and frame_idx not in self.exemplar_frames:
            self.history.append(Sample(frame_idx, self.video.frame_paths[frame_idx], sam3_box))
        self._update_occlusion(occluded, sam3_box)
        record.is_occluded = occluded
        record.occluded_frames = self.occluded_frames
        record.frames_since_reappearance = self.frames_since_reappearance

        correction = None
        if self.vote is not None:
            if frame_idx in self.vote.frames:
                correction = self._cast_vote(record, sam3_box, eff_iou)
        else:
            correction = self._check_trigger(record, sam3_box, eff_iou, occluded)
        self.was_occluded = occluded
        return correction

    def _update_occlusion(self, occluded: bool, sam3_box):
        if occluded and not self.was_occluded and sam3_box is None and self.history:
            self.anchor = self.history[-1]
        if occluded:
            self.occluded_frames += 1
        else:
            self.occluded_frames = 0
            self.anchor = None
        if self.was_occluded and not occluded:
            self.frames_since_reappearance = 0
        else:
            self.frames_since_reappearance += 1

    def _trigger_reason(self, eff_iou, occluded: bool) -> Optional[str]:
        args = self.args
        if not occluded and self.frames_since_reappearance == args.reappearance_cooldown:
            reason = "reappear"
        elif occluded and self.occluded_frames % args.occlusion_poll_interval == 0:
            reason = "occ_poll"
        elif self.streaming is not None:
            last_fire = self.streaming.last_fire_t
            reason = "streaming_var" if self.streaming.step(eff_iou if eff_iou is not None else float("nan")) else None
        else:
            reason = None
        if reason is not None and eff_iou is not None and eff_iou > args.assessor_max_eff_iou:
            # SAM 3 is confident: no call, and a streaming fire does not count for spacing.
            if reason == "streaming_var":
                self.streaming.last_fire_t = last_fire
            return None
        return reason

    def _check_trigger(self, record: FrameRecord, sam3_box, eff_iou, occluded: bool) -> Optional[Correction]:
        reason = self._trigger_reason(eff_iou, occluded)
        if reason is None or not self.exemplars or len(self.history) < 2:
            return None
        frame_idx = record.frame_idx
        context = build_context(self.exemplars, self.history, self.anchor, frame_idx,
                                reason == "occ_poll", self.args)
        query_image = open_rgb(self.video.frame_paths[frame_idx])
        verdict = self.agents.assess(context, query_image, sam3_box, self.video.seq_id, self.obj_id, frame_idx)
        record.vlm_triggered = True
        record.trigger_reason = reason
        record.assessor_verdict = verdict
        record.vote_verdicts = [verdict]
        if verdict != "INCORRECT":
            return None

        vote_frames = {frame_idx + self.args.vote_stride * i for i in range(1, self.args.vote_count)}
        vote_frames = {f for f in vote_frames if f < self.num_frames}
        if vote_frames:
            self.vote = PendingVote(vote_frames, ["INCORRECT"], context, reason)
            return None
        # No frames left to vote on (end of the video): correct now.
        return self._run_corrector(record, context, query_image, sam3_box, eff_iou)

    def _cast_vote(self, record: FrameRecord, sam3_box, eff_iou) -> Optional[Correction]:
        vote, frame_idx = self.vote, record.frame_idx
        query_image = open_rgb(self.video.frame_paths[frame_idx])
        vote.verdicts.append(self.agents.assess(vote.context, query_image, sam3_box,
                                                self.video.seq_id, self.obj_id, frame_idx))
        vote.frames.discard(frame_idx)

        cast, remaining = len(vote.verdicts), len(vote.frames)
        incorrect = vote.verdicts.count("INCORRECT")
        majority = (cast + remaining) // 2
        if incorrect > majority:
            self.vote = None
            record.vlm_triggered = True
            record.trigger_reason = vote.trigger_reason
            record.assessor_verdict = "INCORRECT"
            record.vote_verdicts = list(vote.verdicts)
            return self._run_corrector(record, vote.context, query_image, sam3_box, eff_iou)
        if incorrect + remaining <= majority or cast >= self.args.vote_count:
            self.vote = None
            record.vote_verdicts = list(vote.verdicts)
        return None

    def _run_corrector(self, record: FrameRecord, context: Context, query_image, sam3_box,
                       eff_iou) -> Optional[Correction]:
        record.corrector_called = True
        box, visible = self.agents.correct(context, query_image, sam3_box, self.video.seq_id, self.obj_id,
                                           record.frame_idx)
        correction = self.agents.correction(record.frame_idx, box, visible, sam3_box, eff_iou)
        if correction is not None:
            record.vlm_bbox = correction.box
            record.vlm_corrected = True
        return correction

    def _commit(self, correction: Correction):
        if correction.box is not None:
            prompt_box(self.tracker, self.state, correction.frame_idx, self.obj_id, correction.box,
                       self.width, self.height)
        else:
            prompt_absent(self.tracker, self.state, correction.frame_idx, self.obj_id, self.width, self.height)
        self.history = [s for s in self.history if s.frame_idx < correction.frame_idx]
        self.was_occluded = correction.box is None
        self.occluded_frames = 0
        self.anchor = None
