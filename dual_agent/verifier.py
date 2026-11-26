"""Qwen3-VL wrapper for the assessor and the corrector.

Exemplar frames are drawn with green boxes, context frames and the query frame
with red boxes. Boxes in answers are in 0-1000 normalised coordinates.
"""

import json
import logging
import math
import os
import re
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from prompts import build_tracking_prompt, load_format_suffix

logger = logging.getLogger(__name__)

GREEN = (0, 255, 0)
RED = (255, 0, 0)
MIN_PIXELS = 3136
MAX_NEW_TOKENS = 4096
# Qwen3 thinking-mode sampling parameters.
THINKING_SAMPLING = dict(do_sample=True, temperature=0.6, top_k=20, top_p=0.95, repetition_penalty=1.05)

_THINK_GUIDELINES = re.compile(
    r"REASONING GUIDELINES \(use these steps in your <think> block\):.*?(?=OUTPUT FORMAT|$)", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def sampling_seed(video_id, obj_id, frame_idx, call) -> int:
    key = f"{video_id}|{obj_id}|{frame_idx}|{call}".encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


@dataclass
class Answer:
    verdict: Optional[str]          # "CORRECT" / "INCORRECT" (assessor only)
    visible: Optional[bool]
    box: Optional[List[int]]        # [x1, y1, x2, y2] in pixels when visible
    raw_output: str


def draw_box(image: np.ndarray, box: List[int], color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return out
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _normalized(box: List[int], width: int, height: int) -> List[int]:
    return [int(round(box[0] * 1000.0 / width)), int(round(box[1] * 1000.0 / height)),
            int(round(box[2] * 1000.0 / width)), int(round(box[3] * 1000.0 / height))]


def _patch_hub_for_local_dirs():
    """Resolve hub cache lookups of local model directories from the directory itself."""
    try:
        import transformers.utils.hub as hub
        if not hasattr(hub, "_get_cache_file_to_return"):
            return
        original = hub._get_cache_file_to_return

        def lookup(repo_id, filename, *args, **kwargs):
            if os.path.isabs(repo_id) and os.path.isdir(repo_id):
                path = os.path.join(repo_id, filename)
                return path if os.path.exists(path) else None
            return original(repo_id, filename, *args, **kwargs)

        hub._get_cache_file_to_return = lookup
    except Exception as e:
        logger.warning(f"Could not patch the transformers hub lookup: {e}")


def load_model(model_path: str, device: str):
    """Load a model directory, or a LoRA adapter directory merged into the base
    model named in its adapter_config.json. Returns (model, processor)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device_map = device if device != "cpu" else None
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map)
        return model, AutoProcessor.from_pretrained(model_path)

    with open(adapter_config) as f:
        base_path = json.load(f)["base_model_name_or_path"]
    logger.info(f"Loading base model {base_path} + LoRA adapter {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=device_map)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    # The adapter directory carries the processor it was trained with, if any.
    has_processor = os.path.isfile(os.path.join(model_path, "tokenizer_config.json"))
    processor = AutoProcessor.from_pretrained(model_path if has_processor else base_path)
    return model, processor


class TrackingVerifier:

    def __init__(self, model_path: str, task: str, format_prompt: str, thinking: bool,
                 seed_sampling: bool = True, resize: int = 1000, max_pixels: int = 401408,
                 device: Optional[str] = None):
        if task not in ("assessment", "correction"):
            raise ValueError(f"unknown task {task!r}")
        path = Path(model_path)
        if not path.is_absolute() and path.exists():
            model_path = str(path.resolve())

        self.task = task
        self.thinking = thinking
        self.resize = resize
        self.max_pixels = max_pixels
        self.seed_sampling = seed_sampling
        self.format_suffix = load_format_suffix(format_prompt)
        if not thinking:
            self.format_suffix = _THINK_BLOCK.sub("<think></think>", _THINK_GUIDELINES.sub("", self.format_suffix))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._identity = None
        self._identity_calls = 0

        if os.path.isdir(model_path):
            _patch_hub_for_local_dirs()
        logger.info(f"Loading {task} model from {model_path}")
        self.model, self.processor = load_model(model_path, self.device)
        if self.device == "cpu":
            self.model = self.model.float()
        self.model.eval()

    def set_sampling_identity(self, video_id, obj_id, frame_idx, call) -> None:
        self._identity = (video_id, obj_id, frame_idx, call)
        self._identity_calls = 0

    def verify(self, context_images: List[Image.Image], context_boxes: List[Optional[List[int]]],
               query_image: Image.Image, query_box: Optional[List[int]], frame_indices: List[int],
               num_exemplars: int, num_uniform: int) -> Answer:
        """One model call. context_images/boxes: exemplars first, then context frames.
        frame_indices: source frame index of every image, query last."""
        width, height = query_image.size
        images = self._overlay(context_images, context_boxes, query_image, query_box, num_exemplars)

        prompt_boxes = [_normalized(b, width, height) if b is not None else None for b in context_boxes]
        has_query_box = query_box is not None and not all(v == 0 for v in query_box)
        prompt_boxes.append(_normalized(query_box, width, height) if has_query_box else None)
        try:
            prompt = build_tracking_prompt(
                num_images=len(images),
                num_first_appearance=num_exemplars,
                video_frame_indices=frame_indices,
                num_uniform=num_uniform,
                frame_bboxes=prompt_boxes,
            ) + self.format_suffix
            raw_output = self._generate(prompt, images)
            logger.info(f"Model output: {raw_output[:500]}")
            return self._parse(raw_output, width, height)
        except Exception as e:
            logger.error(f"Verification inference failed: {e}")
            traceback.print_exc()
            return Answer("INCORRECT", None, None, f"Error: {e}")

    def _scaled(self, image: Image.Image, box: Optional[List[int]]):
        """Downscale so the long side is at most `resize`; scale the box with it."""
        w, h = image.size
        scale = self.resize / max(w, h)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            if box is not None:
                box = [int(v * scale) for v in box]
        return image, box

    def _overlay(self, context_images, context_boxes, query_image, query_box, num_exemplars):
        images = []
        for i, (image, box) in enumerate(zip(context_images, context_boxes)):
            image, box = self._scaled(image, box)
            pixels = np.array(image)
            if box is not None:
                pixels = draw_box(pixels, box, GREEN if i < num_exemplars else RED)
            images.append(Image.fromarray(pixels))

        query, box = self._scaled(query_image.copy(), list(query_box) if query_box else None)
        if box is not None and not all(v == 0 for v in box):
            query = Image.fromarray(draw_box(np.array(query), box, RED))
        images.append(query)
        return images

    def _fit_pixel_budget(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        area = image.width * image.height
        if area > self.max_pixels:
            s = math.sqrt(self.max_pixels / area)
            image = image.resize((int(image.width * s), int(image.height * s)))
        area = image.width * image.height
        if area < MIN_PIXELS:
            s = math.sqrt(MIN_PIXELS / area)
            image = image.resize((int(image.width * s), int(image.height * s)))
        return image

    def _generate(self, prompt: str, images: List[Image.Image]) -> str:
        content = []
        for i, text in enumerate(prompt.split("<image>")):
            if i:
                content.append({"type": "image"})
            if text:
                content.append({"type": "text", "text": text})
        template_kwargs = {} if self.thinking else {"enable_thinking": False}
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **template_kwargs)
        inputs = self.processor([self._fit_pixel_budget(im) for im in images], [text],
                                add_special_tokens=False, return_tensors="pt", padding=True).to(self.device)

        gen_kwargs = dict(use_cache=True, max_new_tokens=MAX_NEW_TOKENS)
        gen_kwargs.update(THINKING_SAMPLING if self.thinking else dict(do_sample=False))

        saved_rng = None
        if self.seed_sampling and gen_kwargs["do_sample"]:
            if self._identity is None:
                raise RuntimeError("seed_sampling is on but set_sampling_identity() was not called")
            video_id, obj_id, frame_idx, call = self._identity
            seed = sampling_seed(video_id, obj_id, frame_idx, f"{call}#{self._identity_calls}")
            self._identity_calls += 1
            saved_rng = (torch.get_rng_state(),
                         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if saved_rng is not None:
                torch.set_rng_state(saved_rng[0])
                if saved_rng[1] is not None:
                    torch.cuda.set_rng_state_all(saved_rng[1])

        prompt_len = len(inputs.input_ids[0])
        return self.processor.decode(output_ids[0][prompt_len:], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def _parse(self, raw_output: str, width: int, height: int) -> Answer:
        """Read the answer JSON from <answer>...</answer>; without the tag, take the
        first JSON object with an answer key after </think>, else anywhere."""
        keys = ("tracking_correct", "visible", "box_2d") if self.task == "assessment" else ("box_2d", "visible")

        def find_json(text: str) -> Optional[str]:
            for key in keys:
                m = re.search(r'\{[^{}]*"' + re.escape(key) + r'"\s*:.*?\}', text, re.DOTALL)
                if m:
                    return m.group(0).strip()
            return None

        def unparsed() -> Answer:
            lower = raw_output.lower()
            visible = False if ('"visible": false' in lower or '"visible":false' in lower) else None
            return Answer(None, visible, None, raw_output)

        tagged = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)
        if tagged:
            answer = tagged.group(1).strip()
        else:
            after_think = re.search(r"</think>\s*(.*)", raw_output, re.DOTALL)
            answer = find_json(after_think.group(1) if after_think else raw_output)
            if answer is None and after_think is not None:
                answer = find_json(raw_output)
            if answer is None:
                logger.warning(f"No answer found ({self.task}); output tail: {raw_output[-200:]!r}")
                return unparsed()
            logger.info(f"No <answer> tag; using JSON ({self.task}): {answer[:200]}")

        try:
            data = json.loads(answer)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse answer JSON: {answer[:200]}")
            return unparsed()

        x_scale, y_scale = width / 1000.0, height / 1000.0

        def to_pixels(box):
            if isinstance(box, list) and len(box) == 4:
                try:
                    return [int(box[0] * x_scale), int(box[1] * y_scale),
                            int(box[2] * x_scale), int(box[3] * y_scale)]
                except (TypeError, ValueError) as e:
                    logger.warning(f"Bad box in answer: {e}")
            return None

        verdict = None
        if self.task == "assessment":
            correct = data.get("tracking_correct")
            if isinstance(correct, bool):
                verdict = "CORRECT" if correct else "INCORRECT"
        visible = data.get("visible")
        if visible not in (True, False):
            visible = None
        box = to_pixels(data.get("box_2d")) if visible else None
        return Answer(verdict, visible, box, raw_output)
