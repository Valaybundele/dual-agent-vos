"""SeCVOS and LVOS-v2 readers and mask writers.

Frame indices are positions in the sorted list of a video's JPEG frames, which
is how SAM 3 indexes the video.

SeCVOS (SA-V layout): JPEGImages/<video>/<frame>.jpg, Annotations/<video>/<obj:03d>/<frame>.png
LVOS-v2 (DAVIS layout): JPEGImages/<video>/<frame>.jpg, Annotations/<video>/<frame>.png (palette, pixel = object id)
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from sam3_tracker import mask_to_box

if TYPE_CHECKING:
    from track import ObjectResult

logger = logging.getLogger(__name__)

Boxes = Dict[int, Dict[int, Optional[List[int]]]]   # frame -> object -> GT box (None: empty mask)


@dataclass
class Video:
    seq_id: str
    frame_paths: List[str]
    gt_boxes: Boxes
    gt_mask: Callable[[int, int], Optional[np.ndarray]]   # (frame, obj) -> binary mask or None
    annotated_frames: Optional[Dict[int, List[int]]] = None   # obj -> frames with a GT mask (SeCVOS)

    @property
    def obj_ids(self) -> List[int]:
        return sorted({obj for objs in self.gt_boxes.values() for obj in objs})


def _frame_paths(frames_root: Path, seq_id: str) -> List[str]:
    return sorted(str(p) for p in (frames_root / seq_id).glob("*.jpg"))


def _read_binary(png: Path) -> np.ndarray:
    mask = np.array(Image.open(png))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


class SeCVOS:
    def __init__(self, root: str):
        self.root = Path(root)
        self.frames_root = self.root / "JPEGImages"
        self.masks_root = self.root / "Annotations"

    def sequence_ids(self) -> List[str]:
        id_file = self.root / "video_id.txt"
        if id_file.exists():
            return sorted(line.strip() for line in id_file.read_text().splitlines() if line.strip())
        return [d.name for d in sorted(self.masks_root.iterdir()) if d.is_dir()]

    def load(self, seq_id: str) -> Optional[Video]:
        frame_paths = _frame_paths(self.frames_root, seq_id)
        if not frame_paths:
            logger.warning(f"No frames for {seq_id}, skipping")
            return None
        ann_dir = self.masks_root / seq_id
        boxes: Boxes = defaultdict(dict)
        annotated: Dict[int, List[int]] = {}
        if ann_dir.exists():
            for obj_dir in sorted(ann_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                obj_id = int(obj_dir.name)
                pngs = sorted(obj_dir.glob("*.png"))
                for png in pngs:
                    boxes[int(png.stem)][obj_id] = mask_to_box(_read_binary(png))
                annotated[obj_id] = sorted(int(p.stem) for p in pngs)
        if not boxes:
            logger.warning(f"No annotations for {seq_id}, skipping")
            return None

        def gt_mask(frame_idx: int, obj_id: int) -> Optional[np.ndarray]:
            png = ann_dir / f"{obj_id:03d}" / f"{frame_idx:05d}.png"
            return _read_binary(png) if png.exists() else None

        return Video(seq_id, frame_paths, boxes, gt_mask, annotated)

    def save_masks(self, video: Video, results: Dict[int, "ObjectResult"], out_dir: str, height: int, width: int):
        """One binary PNG per object and GT-annotated frame (empty where SAM 3 has no mask)."""
        for obj_id, result in results.items():
            obj_dir = os.path.join(out_dir, video.seq_id, f"{obj_id:03d}")
            os.makedirs(obj_dir, exist_ok=True)
            if obj_id in video.annotated_frames:
                frames = video.annotated_frames[obj_id]
            elif result.pred_masks:
                frames = sorted(result.pred_masks)
            else:
                continue
            for frame_idx in frames:
                mask = result.pred_masks.get(frame_idx)
                if mask is None:
                    mask = np.zeros((height, width), dtype=np.uint8)
                elif mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").convert("1")
                image.save(os.path.join(obj_dir, f"{frame_idx:05d}.png"))


def _voc_palette() -> List[int]:
    palette = []
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (c & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette.extend([r, g, b])
    return palette


class LVOS:
    """LVOS-v2; root is the split directory (e.g. LVOSv2/valid)."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.frames_root = self.root / "JPEGImages"
        self.masks_root = self.root / "Annotations"

    def sequence_ids(self) -> List[str]:
        return sorted(d.name for d in self.frames_root.iterdir() if d.is_dir())

    def load(self, seq_id: str) -> Optional[Video]:
        frame_paths = _frame_paths(self.frames_root, seq_id)
        if not frame_paths:
            logger.warning(f"No frames for {seq_id}, skipping")
            return None
        ann_dir = self.masks_root / seq_id
        if not ann_dir.exists():
            logger.warning(f"No annotations for {seq_id}, skipping")
            return None
        frame_of_stem = {Path(p).stem: i for i, p in enumerate(frame_paths)}
        boxes: Boxes = defaultdict(dict)
        for png in sorted(ann_dir.glob("*.png")):
            frame_idx = frame_of_stem.get(png.stem)
            if frame_idx is None:
                continue
            labels = np.array(Image.open(png))
            for obj_id in np.unique(labels):
                if obj_id != 0:
                    boxes[frame_idx][int(obj_id)] = mask_to_box((labels == obj_id).astype(np.uint8))
        if not boxes:
            logger.warning(f"No annotations for {seq_id}, skipping")
            return None

        def gt_mask(frame_idx: int, obj_id: int) -> Optional[np.ndarray]:
            png = ann_dir / f"{Path(frame_paths[frame_idx]).stem}.png"
            if not png.exists():
                return None
            mask = (np.array(Image.open(png)) == obj_id).astype(np.uint8)
            return mask if mask.any() else None

        return Video(seq_id, frame_paths, dict(boxes), gt_mask)

    def save_masks(self, video: Video, results: Dict[int, "ObjectResult"], out_dir: str, height: int, width: int):
        """One palette PNG per frame with any predicted mask; higher object ids are drawn on top."""
        seq_dir = os.path.join(out_dir, video.seq_id)
        os.makedirs(seq_dir, exist_ok=True)
        palette = _voc_palette()
        frames = sorted({f for result in results.values() if result.pred_masks for f in result.pred_masks})
        for frame_idx in frames:
            labels = np.zeros((height, width), dtype=np.uint8)
            for obj_id in sorted(results):
                mask = (results[obj_id].pred_masks or {}).get(frame_idx)
                if mask is None:
                    continue
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                labels[mask > 0] = obj_id
            image = Image.fromarray(labels, mode="P")
            image.putpalette(palette)
            image.save(os.path.join(seq_dir, f"{Path(video.frame_paths[frame_idx]).stem}.png"))


DATASETS = {"secvos": SeCVOS, "lvos": LVOS}
