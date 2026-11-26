"""What the two VLM agents see and do on one trigger."""

import logging
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

from PIL import Image

from sam3_tracker import box_iou

logger = logging.getLogger(__name__)


class Sample(NamedTuple):
    frame_idx: int
    path: str
    box: Optional[List[int]]


@dataclass
class Context:
    """Frames shown before the query frame: GT exemplars, earlier context, recent context."""
    images: List[Image.Image]
    boxes: List[Optional[List[int]]]
    frame_indices: List[int]
    num_exemplars: int
    num_uniform: int


@dataclass
class Correction:
    frame_idx: int
    box: Optional[List[int]]   # None: the object is not visible


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_context(exemplars: List[Sample], history: List[Sample], anchor: Optional[Sample],
                  query_idx: int, occlusion_poll: bool, args) -> Context:
    """Exemplars, `num_uniform` frames spread over the older history and the most
    recent history frames. While the object is lost, the last frame before it was
    lost (`anchor`) is kept among the recent frames."""
    if occlusion_poll:
        num_exemplars, num_recent, num_uniform = min(2, len(exemplars)), 2, 2
    else:
        num_exemplars = min(args.num_first_appearance, len(exemplars))
        num_recent, num_uniform = args.window_size - 1, args.num_uniform

    before = [s for s in history if s.frame_idx < query_idx]
    if anchor is not None and anchor.frame_idx < query_idx:
        recent = [s for s in before if s.frame_idx != anchor.frame_idx][-(num_recent - 1):] + [anchor]
        recent.sort(key=lambda s: s.frame_idx)
    else:
        recent = before[-num_recent:]

    window_start = recent[0].frame_idx if recent else query_idx
    older = [s for s in history if s.frame_idx < window_start]
    if len(older) >= num_uniform:
        step = len(older) / num_uniform
        uniform = [older[int(i * step)] for i in range(num_uniform)]
    else:
        uniform = list(older)

    shown = exemplars[:num_exemplars] + uniform + recent
    return Context(
        images=[open_rgb(s.path) for s in shown],
        boxes=[s.box for s in shown],
        frame_indices=[s.frame_idx for s in shown],
        num_exemplars=num_exemplars,
        num_uniform=len(uniform),
    )


class DualAgent:
    def __init__(self, assessor, corrector, args):
        self.assessor = assessor
        self.corrector = corrector
        self.args = args

    def _ask(self, agent, context: Context, query_image, query_box, identity) -> "Answer":
        agent.set_sampling_identity(*identity)
        return agent.verify(context.images, context.boxes, query_image, query_box,
                            context.frame_indices + [identity[2]],
                            num_exemplars=context.num_exemplars, num_uniform=context.num_uniform)

    def assess(self, context: Context, query_image, query_box, video_id, obj_id, frame_idx) -> Optional[str]:
        return self._ask(self.assessor, context, query_image, query_box,
                         (video_id, obj_id, frame_idx, "assess")).verdict

    def correct(self, context: Context, query_image, query_box, video_id, obj_id,
                frame_idx) -> Tuple[Optional[List[int]], Optional[bool]]:
        """Ask the corrector for a box; the assessor re-checks each proposal and a
        rejected proposal becomes the box the corrector starts from next round."""
        box, visible = None, None
        start_box = query_box
        rounds = self.args.max_correction_rounds
        for k in range(rounds):
            answer = self._ask(self.corrector, context, query_image, start_box,
                               (video_id, obj_id, frame_idx, f"corr{k}"))
            box, visible = answer.box, answer.visible
            if box is None or k == rounds - 1:
                break
            recheck = self._ask(self.assessor, context, query_image, box,
                                (video_id, obj_id, frame_idx, f"recheck{k}"))
            if recheck.verdict == "CORRECT":
                break
            start_box = box
        return box, visible

    def correction(self, frame_idx: int, box, visible, sam3_box, eff_iou) -> Optional[Correction]:
        """Turn the corrector's answer into a SAM 3 prompt, or nothing.

        A box is used only if it differs from SAM 3's; "not visible" is used only
        if SAM 3 has a box and is not confident in it."""
        if box is not None:
            overlap = box_iou(box, sam3_box) if sam3_box else 0.0
            return Correction(frame_idx, box) if overlap < self.args.correction_iou_threshold else None
        if visible is False and sam3_box is not None:
            if eff_iou is not None and eff_iou > self.args.neg_prompt_max_eff_iou:
                logger.debug(f"f{frame_idx}: SAM 3 confident (eff-IoU {eff_iou:.3f}), 'not visible' ignored")
                return None
            return Correction(frame_idx, None)
        return None
