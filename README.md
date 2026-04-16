# SID Learning Platform: Socratic Interdisciplinary Dialogue

Train collaborative LLMs for interdisciplinary Socratic tutoring using multiturn-aware reward optimization and DPO.

<div align="left">

[![](https://img.shields.io/badge/Framework-CollabLLM-purple?style=plastic&logo=Google%20Chrome)](https://github.com/microsoft/collabllm)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.00640)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview

This project applies the **CollabLLM** framework to train models for **Socratic Interdisciplinary Dialogue (SID)** — teaching through iterative questions that guide learners across multiple knowledge domains.

**Key Components:**
- **Interdisciplinary Topics**: Cross-domain question generation with multiturn dialogue
- **Socratic Teaching Quality**: LLM-based pedagogical annotation (8 weighted dimensions)
- **Multiturn-Aware Rewards**: Compute interaction quality across full conversation context
- **DPO Training**: Direct preference optimization for collaborative response generation

## Installation

```bash
conda create -n sid python=3.11
conda activate sid
uv pip install -e .
```

**Optional: For distributed training**
```bash
uv pip install deepspeed
```

# Quick Start

### 1. Setup Environment
```bash
cp .env.example .env
# Fill in your API keys and model paths in .env
```

### 2. Generate Interdisciplinary Dialogue Data
```bash
python scripts/engine/build_dataset.py \
    --dataset_name interdisciplinary \
    --num_samples 500 \
    --output_dir outputs/my_experiment/data
```

### 3. Annotate with Teaching Quality Metrics
```bash
python scripts/benchmark/annotation.py \
    --input-dir outputs/my_experiment/data \
    --output-dir outputs/my_experiment/annotated
```

### 4. Evaluate & Train
```bash
# Objective evaluation (quality metrics)
python scripts/benchmark/objective_eval.py \
    outputs/my_experiment/annotated/*.jsonl \
    --summary-file outputs/my_experiment/metrics.json

# DPO training
python scripts/train/offline_dpo_unsloth.py \
    --data_path outputs/my_experiment/annotated \
    --output_dir outputs/my_experiment/models
```

## Project Structure

```
SID Learning Platform/
├── collabllm/                      # Core framework
│   ├── synthetic.py                # Data generation orchestration
│   ├── simulation.py               # Chat session simulation
│   ├── reward.py                   # Multiturn-aware reward computation
│   ├── metrics/teaching_quality.py # Socratic teaching quality scoring
│   └── prompts/                    # Socratic dialogue prompts
│
├── scripts/                        # Production pipeline
│   ├── build_dpo_dataset.sh        # Full workflow orchestration
│   ├── engine/build_dataset.py     # Dialogue generation
│   ├── benchmark/
│   │   ├── multi_dialogue.py       # Socratic dialogue generation
│   │   ├── annotation.py           # Quality annotation
│   │   └── objective_eval.py       # Metrics computation
│   └── train/offline_dpo_unsloth.py # Model training
│
├── data/                           # Datasets
│   ├── interdisciplinary_topic.json
│   ├── multi_dialogue/             # Generated dialogues
│   └── annotated/                  # Annotated conversations
│
└── outputs/                        # Models & results
    └── dpo_*/                      # Experiment outputs
```

## Data Format

**Input**: `data/interdisciplinary_topic.json`
```json
[
  {
    "topic": "How does photosynthesis relate to climate change?",
    "discipline_a": "Biology",
    "discipline_b": "Environmental Science"
  }
]
```

**Generated Dialogue**: Multi-turn conversation with teacher questions + student responses

**Teaching Quality Metrics** (8 dimensions):
- StrategyDensity: Question pedagogical variety
- StrategyVariety: Strategy types covered
- IKT: Interdisciplinary knowledge transfer
- BloomProgression: Cognitive level progression
- StructureCompleteness: Dialogue coherence
- L3GuidanceRate: Effective hints/guidance
- CognitiveCorrectionRate: Error handling quality
- ResponseRelevance: Answer appropriateness


## Citation

If you use this project, please cite CollabLLM:

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and Gavin Li and Yao Dou and Weixin Cai and James Zou and Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```
