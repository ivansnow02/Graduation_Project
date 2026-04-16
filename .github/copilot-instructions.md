# Copilot Instructions — SID Learning Platform

**Environment**: macOS, fish shell, uv package manager
**Python**: 3.11 (per `pyproject.toml`)
**GPU**: Required for annotation (vLLM) and training (DPO)

## Project Mission

Train collaborative LLMs for **Socratic Interdisciplinary Dialogue (SID)** — teaching through iterative questioning across multiple knowledge domains. Single dataset + single metric focused on pedagogical quality.

## Core Architecture

| Component              | Location                                                                  | Purpose                                                                                           |
| ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Reward engine**      | `collabllm/reward.py`                                                     | Compute multiturn-aware reward (quality of interaction across full conversation)                  |
| **Dialogue simulator** | `collabllm/simulation.py`                                                 | Run ChatSessionSimulator with teacher/student roles; supports vLLM + OpenAI-compatible backends   |
| **Teaching quality**   | `collabllm/metrics/teaching_quality.py`                                   | LLM-based annotation: 8 weighted dimensions (strategy density, IKT rate, Bloom progression, etc.) |
| **Data generation**    | `scripts/engine/build_dataset.py` + `scripts/benchmark/multi_dialogue.py` | Generate multiturn dialogues from interdisciplinary topics                                        |
| **Annotation**         | `scripts/benchmark/annotation.py`                                         | Extract 9 structured fields per turn using vLLM (local GPU model); resumable                      |
| **Evaluation**         | `scripts/benchmark/objective_eval.py`, `subjective_eval.py`               | Score dialogues: objective (8 metrics) + subjective (judge model)                                 |
| **Training**           | `scripts/train/offline_dpo_unsloth.py`                                    | DPO training using Unsloth (LoRA acceleration) + TRL trainer                                      |

**Data Flow**:

```
data/interdisciplinary_topic.json
    ↓ multi_dialogue.py (8 workers, OpenAI API)
    ↓ multi_dialogue_topic_<ID>.json (nested structure)
    ↓ annotation.py (vLLM, +annotations field)
    ↓ *.jsonl (row = full dialogue + metadata)
    ├→ objective_eval.py (8 metrics → JSON)
    ├→ subjective_eval.py (judge API → JSON)
    └→ offline_dpo_unsloth.py (DPO training)
```

## Key Files & When to Edit

| File                                                  | Purpose                                      | Edit when...                                                      |
| ----------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| `collabllm/prompts/system_prompt_socratic_strict.txt` | Teacher persona & Socratic method            | Changing teaching style or question strategy                      |
| `collabllm/utils/metrics.py`                          | 8-dimension metric calculation               | Adjusting weights or adding new pedagogical dimensions            |
| `scripts/benchmark/multi_dialogue.py`                 | Dialogue generation orchestration            | Changing num_turns, num_candidates, or generation backend         |
| `scripts/benchmark/annotation.py`                     | vLLM-based extraction of 9 fields            | Modifying structured annotation schema or extraction logic        |
| `scripts/benchmark/objective_eval.py`                 | Metric formulas (StrategyDensity, IKT, etc.) | Adding/removing objective metrics                                 |
| `scripts/train/offline_dpo_unsloth.py`                | DPO training loop + hyperparameters          | Tuning learning rate, LoRA rank, batch size, or output dir naming |
| `.env`                                                | API keys, model endpoints, paths             | Setting up new backends (local vLLM, remote judge API, LM Studio) |

## Environment Setup

**Required `.env` keys**:

```bash
# Data generation (Socratic dialogue)
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"  # or local LM Studio: http://localhost:1234/v1

# Annotation (local vLLM)
ANNOTATION_MODEL="Qwen/Qwen2.5-14B-Instruct"
ANNOTATION_TOKENIZER="Qwen/Qwen2.5-14B-Instruct"
LM_STUDIO_API_URL="http://localhost:1234/v1"  # vLLM endpoint for annotation

# Evaluation (judge model)
JUDGE_API_URL="https://api.deepseek.com/v1"  # or local endpoint
JUDGE_MODEL="deepseek-v3"

# Paths
OUTPUT_DIR="outputs/"
BASE_PATH="/path/to/huggingface/cache"
```

## Standard Commands

```bash
# Full pipeline (500 samples, 3 candidates, experiment = dpo_exp1, judge = qwen)
./scripts/build_dpo_dataset.sh all dpo_exp1 qwen-flash

# Step by step
# 1. Generate (12 hours for 5k samples on GPU)
uv run scripts/engine/build_dataset.py --dataset_name interdisciplinary --num_samples 500

# 2. Annotate (vLLM local, ~4-8 hours for 500 × 3 candidates)
uv run scripts/benchmark/annotation.py --input-dir outputs/dpo_exp1/data --output-dir outputs/dpo_exp1/annotated

# 3. Evaluate
uv run scripts/benchmark/objective_eval.py outputs/dpo_exp1/annotated/*.jsonl
uv run scripts/benchmark/subjective_eval.py --data-dir outputs/dpo_exp1/annotated --judge-model qwen

# 4. Train (Unsloth DPO, ~2-4 hours on single GPU)
uv run scripts/train/offline_dpo_unsloth.py --data_path outputs/dpo_exp1/annotated \
    --output_dir outputs/dpo_exp1_models --model_name Qwen/Qwen3-14B
```

## Project Conventions

**Naming**:

- Experiment directory: `outputs/dpo_<name>_<variant>/`
- Generated dialogues: `multi_dialogue_topic_<ID>.json` (temp, can delete post-annotation)
- Annotated data: `*.jsonl` (1 dialogue per line, +`annotations` field)
- Trained model: `outputs/dpo_<name>_opt/` with checkpoint subdirs

**Annotation format** (9 structured fields per turn):

```json
{
  "speaker": "teacher" | "student",
  "discipline": "Biology" | "Chemistry" | ...,
  "bloom_level": 1-6,
  "pedagogical_strategy": "direct_question" | "hint" | ...,
  "discipline_transfer": true | false,
  "correctness": true | false | "partial",
  "guidance_effectiveness": 0-1,
  "response_relevance": 0-1,
  "explanation_clarity": 0-1
}
```

**Objective metrics** (8 dimensions, returned as weighted TotalScore):

- StrategyDensity (question variety per turn)
- StrategyVariety (types of strategies covered)
- IKT (interdisciplinary knowledge transfer rate)
- BloomProgression (cognitive level climb)
- StructureCompleteness (dialogue coherence)
- L3GuidanceRate (effective guidance (%))
- CognitiveCorrectionRate (error handling)
- ResponseRelevance (student answer appropriateness)

**Resumable jobs**: All batch scripts check existing output line count to resume. **Never reorder JSONL files.**

## Common Pitfalls

| Pitfall                                 | Fix                                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| vLLM OOM during annotation              | Reduce `max_workers` (default 4) or use smaller model (7B instead of 14B)                    |
| `BASE_URL` trailing slash               | Must be `http://localhost:8000/v1` not `http://localhost:8000` or `http://localhost:8000/`   |
| Judge API timeouts                      | Set `JUDGE_API_TIMEOUT=120` in `.env`; check `JUDGE_API_URL` connectivity                    |
| Duplicate candidates in generation      | Ensure `temperature=1.0` and different random seeds per candidate                            |
| Annotation resumes but produces garbage | Check that model weights/tokenizer path are correct (ANNOTATION_TOKENIZER mismatch)          |
| DPO training loss stays flat            | Verify `.jsonl` contains valid `annotations` field; check learning rate (3e-4 typical)       |
| Python version error                    | Use exactly Python 3.11 (per pyproject.toml); 3.13 not supported due to dependency conflicts |

## Adding Features

**New pedagogical dimension** → `collabllm/utils/metrics.py` (add to 8-dimension dict weight) + `collabllm/prompts/` (add extraction prompt)

**New evaluation metric** → `scripts/benchmark/objective_eval.py` (add formula & weight)

**Change Socratic style** → `collabllm/prompts/system_prompt_*.txt` (modify teacher persona)

**Integrate new judge model** → Update `JUDGE_MODEL` in `.env`; test JSON parsing fallback in `scripts/benchmark/subjective_eval.py`

## Key Dependencies

- **Unsloth** (DPO acceleration): Requires torch + transformers; install torch first
- **vLLM** (local annotation): GPU required; 16GB VRAM for 14B in 4-bit
- **litellm** (LLM backend abstraction): Dynamically imports backends; may fail at runtime if dependency missing
- **json-repair** (malformed JSON fix): Best-effort; some formats still fail parsing

## Links

- [README](../../README.md) — Project overview & quick start
- [pyproject.toml](../../pyproject.toml) — Dependencies & Python version
- Copilot instructions follow: UV package manager, fish shell, single-GPU focus
