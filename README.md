# When Trackers Fail: VLM-Guided Verification and Recovery for Robust Video Object Segmentation

**NeurIPS 2026** · [Paper](https://arxiv.org/abs/2610.15929) · [Corrector weights](https://huggingface.co/valayy03/dual-agent-vos-corrector)

Valay Bundele, Susmit Agrawal, Mehran Hosseinzadeh, The Nam Nguyen, Hendrik P. A. Lensch

Official evaluation code for **Dual-Agent** on SeCVOS and LVOS-v2.

Dual-Agent adds two vision-language agents to SAM 3. An **assessor** checks the track at
trigger frames (reappearance after occlusion, periodic polling during occlusion, and unstable
SAM 3 confidence). When the assessor rejects the track, a **corrector** predicts the target box
or reports that the target is not visible, and SAM 3 resumes tracking from that prompt.

| Component | Model |
|---|---|
| Tracker | [SAM 3](https://github.com/facebookresearch/sam3) |
| Assessor | [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) (zero-shot) |
| Corrector | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) + [LoRA](https://huggingface.co/valayy03/dual-agent-vos-corrector) |

## Results

| Method | SeCVOS J&F | LVOS-v2 val J&F |
|---|---|---|
| SAM 3 | 74.21 | 88.2 |
| Dual-Agent | **76.94** | **89.4** |

## Installation

```bash
git clone https://github.com/facebookresearch/sam3
git -C sam3 checkout 11dec29
pip install -e sam3
pip install -r requirements.txt
```

Download the models:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Thinking --local-dir models/Qwen3-VL-8B-Thinking
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
huggingface-cli download valayy03/dual-agent-vos-corrector --local-dir models/corrector-lora
```

SAM 3 weights are fetched from [facebook/sam3](https://huggingface.co/facebook/sam3) on first use.

Download [SeCVOS](https://huggingface.co/datasets/OpenIXCLab/SeCVOS) and [LVOS-v2](https://github.com/LingyiHongfd/LVOS).

## Evaluation

```bash
export ASSESSOR_MODEL=models/Qwen3-VL-8B-Thinking
export CORRECTOR_ADAPTER=models/corrector-lora
```

Predicted masks are written to `$OUTPUT_DIR/Annotations`. Set `NUM_GPUS` to split the videos
across GPUs.

### SeCVOS

```bash
export SECVOS_ROOT=/path/to/SeCVOS
OUTPUT_DIR=out/secvos bash scripts/run_secvos.sh
```

Score with the official [SeC](https://github.com/OpenIXCLab/SeC) evaluator:

```bash
git clone https://github.com/OpenIXCLab/SeC
SEC_REPO=SeC PRED_DIR=out/secvos/Annotations bash scripts/score_secvos.sh
```

### LVOS-v2

```bash
export LVOS_ROOT=/path/to/LVOSv2/valid
OUTPUT_DIR=out/lvos bash scripts/run_lvos.sh
```

Score with the official [LVOS](https://github.com/LingyiHongfd/lvos-evaluation) evaluator:

```bash
git clone https://github.com/LingyiHongfd/lvos-evaluation
LVOS_EVAL=lvos-evaluation PRED_DIR=out/lvos/Annotations bash scripts/score_lvos.sh
```

## Citation

```bibtex
@inproceedings{bundele2026trackersfail,
  title     = {When Trackers Fail: {VLM}-Guided Verification and Recovery for Robust Video Object Segmentation},
  author    = {Bundele, Valay and Agrawal, Susmit and Hosseinzadeh, Mehran and Nguyen, The Nam and Lensch, Hendrik P. A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

[CC BY-NC 4.0](LICENSE). SAM 3, Qwen3-VL, SeCVOS and LVOS are subject to their own licenses.
