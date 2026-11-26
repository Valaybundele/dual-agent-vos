"""Prompt construction for the assessor and the corrector."""

import re
from pathlib import Path
from typing import List, Optional

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_format_suffix(filename: str) -> str:
    """Return the part of a prompt file that follows its leading Jinja placeholder."""
    content = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    marker = "{{ content | trim }}"
    idx = content.find(marker)
    if idx >= 0:
        content = content[idx + len(marker):]
    else:
        m = re.match(r"\s*\{\{[^}]*\}\}", content)
        if m:
            content = content[m.end():]
    # Drop a trailing Jinja fragment (`----` line followed by `{{ ... }}`), if any.
    content = re.sub(r"\n*-+\s*\n+\{\{[^}]*\}\}\s*\Z", "", content)
    return content


def build_tracking_prompt(
    num_images: int,
    num_first_appearance: int = 1,
    video_frame_indices: Optional[List[int]] = None,
    num_uniform: int = 0,
    frame_bboxes: Optional[List[Optional[List]]] = None,
) -> str:
    """Describe the image sequence: GT exemplars, earlier/recent context, query.

    Args:
        num_images: number of images (exemplars + context + query).
        num_first_appearance: number of leading GT exemplar frames.
        video_frame_indices: source video frame index of each image, shown as (t=N).
        num_uniform: number of context frames labelled "earlier context"; the
            remaining context frames are labelled "recent context".
        frame_bboxes: per-image box in 0-1000 normalised xyxy, or None.
    """
    image_tokens = "<image>" * num_images

    def _t(prompt_idx: int) -> str:
        if video_frame_indices and prompt_idx < len(video_frame_indices):
            return f" (t={video_frame_indices[prompt_idx]})"
        return ""

    def _box(prompt_idx: int) -> str:
        if frame_bboxes and prompt_idx < len(frame_bboxes) and frame_bboxes[prompt_idx] is not None:
            b = frame_bboxes[prompt_idx]
            return f" [{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}]"
        return ""

    lines = []

    for i in range(num_first_appearance):
        frame_num = i + 1
        box_str = _box(i)
        if i == 0 and num_first_appearance == 1:
            lines.append(
                f"- Frame {frame_num}{_t(i)}: GROUND TRUTH (GREEN box{box_str}) — "
                "the TARGET OBJECT's definitive appearance."
            )
        else:
            lines.append(
                f"- Frame {frame_num}{_t(i)}: GROUND TRUTH (GREEN box{box_str})"
            )

    context_start_idx = num_first_appearance
    context_end_idx = num_images - 2  # last image is the query
    num_context = context_end_idx - context_start_idx + 1

    for ci in range(num_context):
        pi = context_start_idx + ci
        frame_num = pi + 1
        box_str = _box(pi)
        if num_uniform > 0 and ci < num_uniform:
            role = "earlier context"
        elif num_uniform > 0:
            role = "recent context"
        else:
            role = "context"
        if box_str:
            lines.append(f"- Frame {frame_num}{_t(pi)} [{role}]: RED box{box_str}")
        else:
            lines.append(f"- Frame {frame_num}{_t(pi)} [{role}]: no box (object occluded or tracker lost)")

    qi = num_images - 1
    box_str = _box(qi)
    if box_str:
        lines.append(f"- Frame {num_images}{_t(qi)} [query]: tracker's current prediction — RED box{box_str}")
    else:
        lines.append(f"- Frame {num_images}{_t(qi)} [query]: no tracker prediction shown")

    return (
        f"{image_tokens}\n"
        "You are analyzing a video object tracking task.\n"
        "The notation (t=N) indicates the source video frame number. "
        "Bounding boxes are in [x1, y1, x2, y2] format, 0-1000 normalized coordinates.\n\n"
        "FRAME SEQUENCE:\n"
        + "\n".join(lines) + "\n\n"
        "TASK:\n"
        "Verify whether the tracker's prediction in the last frame is correct. "
        "Determine if the TARGET OBJECT is visible and if the RED box (if any) accurately tracks it. "
        "If the tracking is incorrect and the object is visible, provide the correct bounding box."
    )
