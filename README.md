# SID Learning Platform: Socratic Interdisciplinary Dialogue

Train and evaluate collaborative LLMs for interdisciplinary Socratic tutoring with multi-turn dialogue generation, teaching-quality annotation, reward-aware data construction, and DPO/SFT fine-tuning.

[中文版](README_CN.md)

<div align="left">

[![](https://img.shields.io/badge/Framework-CollabLLM-purple?style=plastic&logo=Google%20Chrome)](https://github.com/microsoft/collabllm)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.00640)
[![Python](https://img.shields.io/badge/Python-3.11--3.12-blue.svg)](https://www.python.org/)
[![Vue](https://img.shields.io/badge/Web_Demo-Vue_3-42b883.svg)](https://vuejs.org/)

</div>

## Overview

This repository implements an end-to-end research and engineering pipeline for **SID (Socratic Interdisciplinary Dialogue)**. It generates multi-turn teacher-student conversations from interdisciplinary topics, annotates teaching quality with structured rubrics, computes objective metrics, converts conversations into preference data, and fine-tunes models with SFT or DPO.

The project follows the spirit of **CollabLLM**: training LLMs to act as active collaborators rather than passive answer generators. In this setting, the assistant is expected to ask useful questions, guide learners through misconceptions, connect concepts across disciplines, and maintain coherent multi-turn instruction.

## Key Features

- **Interdisciplinary topic generation** for cross-domain learning tasks.
- **Multi-turn Socratic tutoring simulation** with teacher questions and student responses.
- **LLM-based teaching-quality annotation** with structured dimensions.
- **Objective evaluation** over conversation-level and file-level metrics.
- **Preference-data construction** from multiple candidate teacher responses.
- **Unsloth-based SFT/DPO training** for Qwen-style models and LoRA adapters.
- **Model merging and GGUF export** for downstream inference tools.
- **Vue 3 web demo** for interactive tutoring, model configuration, evaluation display, and annotation editing.

## Repository Layout

```text
.
├── collabllm/                         # Core Python package
│   ├── synthetic.py                   # Synthetic multi-turn data generation
│   ├── simulation.py                  # Dialogue simulation
│   ├── reward.py                      # Multi-turn reward and preference logic
│   ├── metric.py                      # Metric wrapper
│   ├── metrics/teaching_quality.py    # SID teaching-quality metrics
│   ├── datasets/                      # Dataset loading, cleaning, conversion
│   ├── modules/                       # LLM collaborator and user simulator
│   └── prompts/                       # Prompt templates
│
├── scripts/
│   ├── build_dpo_dataset.sh           # build/train/eval/all experiment driver
│   ├── run_benchmark.sh               # batch annotation and evaluation driver
│   ├── engine/                        # generation and inference scripts
│   ├── benchmark/                     # annotation and objective evaluation
│   ├── converter/                     # annotated data to SFT/DPO formats
│   ├── data_prep/                     # external data preparation
│   └── train/                         # SFT, DPO, LoRA merge scripts
│
├── web_demo/frontend/                 # Vue 3 + Vite demo app
├── pyproject.toml                     # Python package metadata
├── requirements.txt                   # frozen dependency snapshot
└── uv.lock                            # uv lock file
```

`data/` and `outputs/` are generated during data preparation, annotation, evaluation, and training. They are not required to exist in a fresh checkout.

## Requirements

- Python `>=3.11, <3.13`
- `uv` for Python environment management
- CUDA-capable GPU for Unsloth, vLLM, SFT, and DPO workloads
- Bun or npm for the web demo
- API keys or local OpenAI-compatible endpoints for generation and annotation

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

To reproduce the frozen dependency snapshot:

```bash
uv pip install -r requirements.txt
```

Training dependencies are GPU- and CUDA-sensitive. If installation fails, verify the compatibility between CUDA, PyTorch, Unsloth, and vLLM first.

## Environment Variables

Scripts load `.env` when available:

```bash
cp .env.example .env  # if the template exists
```

Typical variables:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
HF_TOKEN=your_huggingface_token
MODELSCOPE_TOKEN=your_modelscope_token
```

Local OpenAI-compatible endpoints can also be passed through CLI JSON arguments or shell variables such as `BASE_URL=http://localhost:8000/v1`.

## Quick Start

Run the integrated DPO dataset pipeline:

```bash
./scripts/build_dpo_dataset.sh build dpo_500_sid
./scripts/build_dpo_dataset.sh train dpo_500_sid
./scripts/build_dpo_dataset.sh eval dpo_500_sid qwen-flash
```

Override defaults:

```bash
BASE_MODEL=/path/to/base-model \
BASE_URL=http://localhost:8000/v1 \
REWARD_MODEL=openai/qwen-flash \
MODEL_NAME=outputs/sid_qwen14b_sft_2500 \
TRAIN_SIZE=500 \
NUM_CANDIDATES=3 \
EPOCHS=3 \
./scripts/build_dpo_dataset.sh all dpo_500_sid qwen-flash
```

Expected outputs:

```text
outputs/<experiment>/interdisciplinary_multiturn.json
outputs/<experiment>_opt/
outputs/logs/<experiment>_*.log
```

## Manual Multi-turn Data Generation

```bash
uv run scripts/engine/build_dataset.py \
  --dataset_name interdisciplinary \
  --metric_names teaching_quality \
  --metric_weights 1.0 \
  --num_candidate_responses 3 \
  --train_size 500 \
  --output_dir outputs/dpo_500_sid \
  --user_prompt_file collabllm/prompts/student_simulator.txt \
  --user_generation_kwargs '{"model":"Qwen/Qwen3-14B","base_url":"http://localhost:8000/v1","api_key":"not-needed","temperature":1.0,"max_tokens":2048}' \
  --assistant_generation_kwargs '{"model":"Qwen/Qwen3-14B","base_url":"http://localhost:8000/v1","api_key":"not-needed","temperature":1.0,"max_tokens":2048}' \
  --reward_generation_kwargs '{"model":"qwen-flash","temperature":0}' \
  --resume
```

## Annotation and Objective Evaluation

Single-machine annotation:

```bash
ANNOTATION_INPUT_DIR=data/dialog/dpo_500_sid \
ANNOTATION_OUTPUT_DIR=data/final/dpo_500_sid \
uv run scripts/benchmark/annotation.py
```

Batch annotation:

```bash
./scripts/run_benchmark.sh prepare dpo_500_sid qwen-flash
# Upload data/batch/dpo_500_sid/batch.jsonl to your batch API provider.
# Download processed JSONL files into data/batchoutput/dpo_500_sid/.
./scripts/run_benchmark.sh eval dpo_500_sid
```

Objective evaluation:

```bash
uv run scripts/benchmark/objective_eval.py \
  data/final/dpo_500_sid/*.jsonl \
  --summary-file data/metrics/dpo_500_sid.json \
  --output-dir data/metrics/dpo_500_sid_per_file_metrics
```

## Training

SFT:

```bash
uv run scripts/train/sft_unsloth.py \
  --dataset_repo data/sft/train.json \
  --output_dir outputs/sid_sft \
  --model_name Qwen/Qwen3-14B
```

DPO:

```bash
uv run scripts/train/offline_dpo_unsloth.py \
  --dataset_repo outputs/dpo_500_sid/interdisciplinary_multiturn.json \
  --output_dir outputs/sid_dpo \
  --model_name outputs/sid_sft \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --max_seq_length 4096 \
  --use_swanlab
```

## Merge and Export

Merge a LoRA adapter into the base model:

```bash
uv run scripts/train/merge_lora_to_base.py \
  --base_model /path/to/base-model \
  --lora_dir outputs/sid_dpo \
  --output_dir outputs/sid_dpo_merged
```

Export GGUF:

```bash
uv run scripts/export_gguf.py \
  --model_dir outputs/sid_dpo_merged \
  --quantization q4_k_m
```

## Web Demo

The frontend lives in `web_demo/frontend`.

```bash
cd web_demo/frontend
bun install
bun run dev
```

Build:

```bash
bun run build
```

Alternative npm workflow:

```bash
npm install
npm run dev
```

The demo supports:

- OpenAI-compatible model configuration
- interactive tutoring conversations
- evaluation-panel display
- structured annotation editing
- JSONL export for current sessions and evaluation results

## Data Format

Example interdisciplinary topic:

```json
[
  {
    "topic": "How does photosynthesis relate to climate change?",
    "discipline_a": "Biology",
    "discipline_b": "Environmental Science"
  }
]
```

Generated data typically includes:

- dialogue context
- candidate teacher responses
- simulated student responses
- teaching-quality scores
- chosen/rejected preference pairs
- metadata for training and evaluation

## Teaching-Quality Metrics

- **StrategyDensity**: density of pedagogical strategies
- **StrategyVariety**: diversity of strategy types
- **IKT**: interdisciplinary knowledge transfer
- **BloomProgression**: progression across Bloom cognitive levels
- **StructureCompleteness**: dialogue coherence and completeness
- **L3GuidanceRate**: effective hints and guidance
- **CognitiveCorrectionRate**: misconception correction quality
- **ResponseRelevance**: response relevance to the learning task

## Publishing Models

This checkout does not include a dedicated model-upload script. Before publishing, decide whether you want to upload the LoRA adapter, the merged full model, or the exported GGUF file, and exclude unnecessary `checkpoint-*` directories. For Hugging Face, use `huggingface-cli upload` or the `huggingface_hub` SDK. For ModelScope, use the official ModelScope SDK.

## Citation

If you use this project, please cite the SID benchmark paper:

```bibtex
@article{jiang2025sid,
    title={SID: Benchmarking Guided Instruction Capabilities in STEM Education with a Socratic Interdisciplinary Dialogues Dataset},
    author={Jiang, Mei and Yue, Houping and Li, Bingdong and Hao, Hao and Qian, Ying and Jiang, Bo and Zhou, Aimin},
    journal={arXiv preprint arXiv:2508.04563},
    year={2025}
}
```

This project also builds on CollabLLM:

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and Gavin Li and Yao Dou and Weixin Cai and James Zou and Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```
