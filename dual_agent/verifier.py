"""Qwen3-VL wrapper for the assessor and the corrector."""

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
from PIL import Image as PILImage

from prompts import build_tracking_prompt, load_format_suffix

logger = logging.getLogger(__name__)

GREEN = (0, 255, 0)
RED = (255, 0, 0)


def compute_sampling_seed(video_id, obj_id, frame_idx, call_index) -> int:
    key = f"{video_id}|{obj_id}|{frame_idx}|{call_index}".encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


@dataclass
class VerificationResult:
    assessment: Optional[str]            # "CORRECT" / "INCORRECT" (assessor), None otherwise
    visible: Optional[bool]              # target visible in the query frame
    predicted_bbox: Optional[List[int]]  # [x1, y1, x2, y2] in pixels when visible
    raw_output: str


def draw_bbox_outline(frame: np.ndarray, bbox: List[int],
                      color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    """Draw a box outline (no fill) on an RGB frame."""
    frame_out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_out
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, thickness)
    return frame_out


class TrackingVerifier:

    def __init__(
        self,
        model_path: str,
        task_schema: str,
        format_prompt: str,
        num_first_appearance: int,
        num_uniform: int,
        nothink: bool,
        seed_sampling: bool = True,
        resize_size: int = 1000,
        max_pixels: int = 401408,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if task_schema not in ("assessment", "correction"):
            raise ValueError(f"unknown task_schema {task_schema!r}")

        # Resolve relative paths so transformers does not read them as hub repo ids.
        _p = Path(model_path)
        if not _p.is_absolute() and _p.exists():
            model_path = str(_p.resolve())

        self.task_schema = task_schema
        self.num_first_appearance = num_first_appearance
        self.num_uniform = num_uniform
        self.nothink = nothink
        self.resize_size = resize_size
        self.max_pixels = max_pixels
        self.min_pixels = 3136
        self.max_new_tokens = 4096
        self.seed_sampling = bool(seed_sampling)
        self._sampling_identity = None   # (video_id, obj_id, frame_idx, call_index)
        self._sampling_subcall = 0
        self.format_suffix = load_format_suffix(format_prompt)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Some transformers versions validate local directories as hub repo ids;
        # answer cache lookups for local directories directly.
        if os.path.isdir(model_path):
            try:
                import transformers.utils.hub as _hub
                if hasattr(_hub, "_get_cache_file_to_return"):
                    _orig_fn = _hub._get_cache_file_to_return

                    def _local_aware(repo_id, filename, *a, **kw):
                        if os.path.isabs(repo_id) and os.path.isdir(repo_id):
                            fp = os.path.join(repo_id, filename)
                            return fp if os.path.exists(fp) else None
                        return _orig_fn(repo_id, filename, *a, **kw)

                    _hub._get_cache_file_to_return = _local_aware
            except Exception as _patch_err:
                logger.warning(f"Could not apply hub patch: {_patch_err}")

        logger.info(f"Loading tracking verification model from {model_path}")
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            # LoRA adapter: load the base model named in adapter_config.json, then merge.
            logger.info(f"Detected LoRA checkpoint at {model_path}")
            with open(adapter_config_path) as f:
                base_model_path = json.load(f)["base_model_name_or_path"]
            logger.info(f"Loading base model: {base_model_path}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device if device != "cpu" else None,
            )
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
            # Processor from the adapter directory when it carries a tokenizer, else from the base.
            if os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                self.processor = AutoProcessor.from_pretrained(base_model_path)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device if device != "cpu" else None,
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

        if device == "cpu":
            self.model = self.model.float()
        self.model.eval()
        logger.info("Model loaded successfully")

    def set_sampling_identity(self, video_id, obj_id, frame_idx, call_index) -> None:
        self._sampling_identity = (video_id, obj_id, frame_idx, call_index)
        self._sampling_subcall = 0

    def _prepare_overlay_images(
        self,
        context_frames: List[PILImage.Image],
        context_bboxes: List[Optional[List[int]]],
        query_frame: PILImage.Image,
        query_bbox: Optional[List[int]],
        num_first_appearance: int,
    ) -> List[PILImage.Image]:
        """Resize each frame to resize_size (long side, downscale only), then draw
        GREEN boxes on exemplars, RED boxes on context frames and the query."""
        images = []
        for i, (frame, bbox) in enumerate(zip(context_frames, context_bboxes)):
            w, h = frame.size
            scale = self.resize_size / max(w, h)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                frame = frame.resize((new_w, new_h), PILImage.BILINEAR)
                if bbox is not None:
                    bbox = [int(v * scale) for v in bbox]
            frame_np = np.array(frame)
            if bbox is not None:
                frame_np = draw_bbox_outline(frame_np, bbox, GREEN if i < num_first_appearance else RED)
            images.append(PILImage.fromarray(frame_np))

        query = query_frame.copy()
        w, h = query.size
        scale = self.resize_size / max(w, h)
        scaled_qbbox = list(query_bbox) if query_bbox else None
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            query = query.resize((new_w, new_h), PILImage.BILINEAR)
            if scaled_qbbox is not None:
                scaled_qbbox = [int(v * scale) for v in scaled_qbbox]
        if scaled_qbbox is not None and not all(v == 0 for v in scaled_qbbox):
            query = PILImage.fromarray(draw_bbox_outline(np.array(query), scaled_qbbox, RED))
        images.append(query)
        return images

    def _run_inference(self, images: List[PILImage.Image],
                       video_frame_indices: Optional[List[int]],
                       num_uniform: int,
                       frame_bboxes: List[Optional[List[int]]],
                       num_first_appearance: int) -> str:
        task_prompt = build_tracking_prompt(
            num_images=len(images),
            num_first_appearance=num_first_appearance,
            video_frame_indices=video_frame_indices,
            num_uniform=num_uniform,
            frame_bboxes=frame_bboxes,
        )
        format_suffix = self.format_suffix
        if self.nothink:
            # No-think decoding: drop the <think> guidelines from the format suffix.
            guidelines_re = re.compile(
                r"REASONING GUIDELINES \(use these steps in your <think> block\):.*?(?=OUTPUT FORMAT|$)",
                re.DOTALL,
            )
            think_re = re.compile(r"<think>.*?</think>", re.DOTALL)
            format_suffix = guidelines_re.sub("", format_suffix)
            format_suffix = think_re.sub("<think></think>", format_suffix)
        prompt_str = task_prompt + format_suffix

        # One user message; <image> placeholders become image entries in order.
        content_list = []
        for i, text_part in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})
            if text_part:
                content_list.append({"type": "text", "text": text_part})
        messages = [{"role": "user", "content": content_list}]

        template_kwargs = {}
        if self.nothink:
            template_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )

        processed = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if (img.width * img.height) > self.max_pixels:
                s = math.sqrt(self.max_pixels / (img.width * img.height))
                img = img.resize((int(img.width * s), int(img.height * s)))
            if (img.width * img.height) < self.min_pixels:
                s = math.sqrt(self.min_pixels / (img.width * img.height))
                img = img.resize((int(img.width * s), int(img.height * s)))
            processed.append(img)

        inputs = self.processor(
            processed,
            [text],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)

        gen_kwargs = dict(use_cache=True, max_new_tokens=self.max_new_tokens)
        if self.nothink:
            gen_kwargs["do_sample"] = False
        else:
            # Qwen3 thinking-mode sampling parameters.
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.6
            gen_kwargs["top_k"] = 20
            gen_kwargs["top_p"] = 0.95
            gen_kwargs["repetition_penalty"] = 1.05

        rng_snapshot = None
        if self.seed_sampling and gen_kwargs["do_sample"]:
            if self._sampling_identity is None:
                raise RuntimeError("seed_sampling is on but set_sampling_identity() was not called")
            vid, oid, fidx, cidx = self._sampling_identity
            seed = compute_sampling_seed(vid, oid, fidx, f"{cidx}#{self._sampling_subcall}")
            self._sampling_subcall += 1
            rng_snapshot = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            )
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if rng_snapshot is not None:
                torch.set_rng_state(rng_snapshot[0])
                if rng_snapshot[1] is not None:
                    torch.cuda.set_rng_state_all(rng_snapshot[1])

        prompt_len = len(inputs.input_ids[0])
        return self.processor.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _parse_output(self, raw_output: str, original_w: int, original_h: int) -> VerificationResult:
        """Read the answer JSON: <answer>...</answer> first, else a schema JSON
        after </think>, else anywhere in the output."""
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)

        if self.task_schema == "assessment":
            schema_keys = ("tracking_correct", "visible", "box_2d")
        else:
            schema_keys = ("box_2d", "visible")

        def _find_schema_json(text: str) -> Optional[str]:
            for key in schema_keys:
                m = re.search(r'\{[^{}]*"' + re.escape(key) + r'"\s*:.*?\}', text, re.DOTALL)
                if m:
                    return m.group(0).strip()
            return None

        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            after_think_m = re.search(r"</think>\s*(.*)", raw_output, re.DOTALL)
            tail = after_think_m.group(1) if after_think_m else raw_output
            answer_content = _find_schema_json(tail)
            if answer_content is None and after_think_m is not None:
                answer_content = _find_schema_json(raw_output)
            if answer_content is not None:
                logger.info(f"No <answer> tag, extracted JSON for "
                            f"schema={self.task_schema}: {answer_content[:200]}")
            else:
                logger.warning(f"No <answer> tag or schema JSON found "
                               f"(schema={self.task_schema}); raw tail: {raw_output[-200:]!r}")
                lower = raw_output.lower()
                if '"visible": false' in lower or '"visible":false' in lower:
                    return VerificationResult(None, False, None, raw_output)
                return VerificationResult(None, None, None, raw_output)

        try:
            data = json.loads(answer_content)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from answer: {answer_content[:200]}")
            lower = raw_output.lower()
            if '"visible": false' in lower or '"visible":false' in lower:
                return VerificationResult(None, False, None, raw_output)
            return VerificationResult(None, None, None, raw_output)

        x_scale = original_w / 1000.0
        y_scale = original_h / 1000.0

        def _scale_bbox(boxes):
            if isinstance(boxes, list) and len(boxes) == 4:
                try:
                    return [int(boxes[0] * x_scale), int(boxes[1] * y_scale),
                            int(boxes[2] * x_scale), int(boxes[3] * y_scale)]
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error scaling bbox: {e}")
            return None

        if self.task_schema == "assessment":
            tracking_correct = data.get("tracking_correct")
            if isinstance(tracking_correct, bool):
                assessment = "CORRECT" if tracking_correct else "INCORRECT"
            else:
                assessment = None  # missing / non-bool: treated as no verdict
            visible = data.get("visible")
            if visible not in (True, False):
                visible = None
            predicted_bbox = _scale_bbox(data.get("box_2d")) if visible else None
        else:
            assessment = None
            visible = data.get("visible")
            if visible not in (True, False):
                visible = None
            predicted_bbox = _scale_bbox(data.get("box_2d")) if visible else None

        return VerificationResult(assessment, visible, predicted_bbox, raw_output)

    def verify(
        self,
        context_frames: List[PILImage.Image],
        context_bboxes: List[Optional[List[int]]],
        query_frame: PILImage.Image,
        query_bbox: Optional[List[int]],
        video_frame_indices: Optional[List[int]] = None,
        num_uniform_override: Optional[int] = None,
        num_first_appearance_override: Optional[int] = None,
    ) -> VerificationResult:
        """Run one call on (exemplars + context frames, query frame).

        context_frames / context_bboxes: exemplars first, then context frames.
        query_bbox: the box to judge (or to start from), None if absent.
        video_frame_indices: source frame index of every image, query last.
        """
        nfa = (num_first_appearance_override if num_first_appearance_override is not None
               else self.num_first_appearance)
        num_uniform = num_uniform_override if num_uniform_override is not None else self.num_uniform
        original_w, original_h = query_frame.size

        images = self._prepare_overlay_images(
            context_frames, context_bboxes, query_frame, query_bbox, nfa)

        # Boxes in 0-1000 normalised coordinates for the prompt text.
        frame_bboxes: List[Optional[List[int]]] = []
        for bbox in context_bboxes:
            if bbox is not None:
                frame_bboxes.append([
                    int(round(bbox[0] * 1000.0 / original_w)),
                    int(round(bbox[1] * 1000.0 / original_h)),
                    int(round(bbox[2] * 1000.0 / original_w)),
                    int(round(bbox[3] * 1000.0 / original_h)),
                ])
            else:
                frame_bboxes.append(None)
        if query_bbox is not None and not all(v == 0 for v in query_bbox):
            frame_bboxes.append([
                int(round(query_bbox[0] * 1000.0 / original_w)),
                int(round(query_bbox[1] * 1000.0 / original_h)),
                int(round(query_bbox[2] * 1000.0 / original_w)),
                int(round(query_bbox[3] * 1000.0 / original_h)),
            ])
        else:
            frame_bboxes.append(None)

        try:
            raw_output = self._run_inference(
                images, video_frame_indices=video_frame_indices, num_uniform=num_uniform,
                frame_bboxes=frame_bboxes, num_first_appearance=nfa)
            logger.info(f"Model output: {raw_output[:500]}")
            return self._parse_output(raw_output, original_w, original_h)
        except Exception as e:
            logger.error(f"Verification inference failed: {e}")
            traceback.print_exc()
            return VerificationResult("INCORRECT", None, None, f"Error: {str(e)}")
