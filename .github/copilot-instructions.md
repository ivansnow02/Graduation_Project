# Copilot Instructions for CollabLLM & SID_Benchmark

## Project Overview

- **CollabLLM**: Framework for training collaborative LLMs with multiturn-aware rewards. Key flows: data generation, annotation, objective/subjective evaluation, and model training.
- **SID_Benchmark**: Scripts for interdisciplinary Socratic dialogue generation, annotation, and evaluation.

## Architecture & Key Components

- `collabllm/`: Core library. Contains reward computation (`reward.py`), simulation, synthetic data, datasets, modules (LLM collaborator, user simulator), prompts, and utilities.
- `scripts/benchmark/`: Benchmark pipeline. Main scripts:
  - `multi_dialogue.py`: Generates Socratic multi-turn dialogues (OpenAI-compatible API, multiprocessing).
  - `annotation.py`: Annotates dialogues with 9 structured fields using vLLM (local model, GPU required).
  - `objective_eval.py`: Computes objective metrics from annotated data (weighted formula, see code).
  - `subjective_eval.py`: Uses judge model API for subjective scoring (5 dimensions, parallelized, JSON output).
- `examples/`: Add new datasets (`single_turn_ds/`) and metrics (`metrics/`). Register in `__init__.py`.
- `.env`: Required for API keys, model paths, and directories. Use provided template and fill real values.

## Developer Workflows

```markdown
# Copilot instructions — quick reference (CollabLLM + SID_Benchmark)

This file gives immediately actionable guidance for AI coding agents working in this repository. Keep edits short and concrete.

- Big picture: Core library `collabllm/` (reward computation, dataset classes, sim/synthetic helpers) + benchmark scripts in `scripts/benchmark/` (generation, annotation, objective/subj eval). Data flow: topics → `multi_dialogue.py` → JSON outputs → `annotation.py` → JSONL with `annotations` → `objective_eval.py` / `subjective_eval.py`.

- Key files to inspect when changing behavior:

  - `scripts/benchmark/multi_dialogue.py` — Socratic generation (teacher-only questions), OpenAI-compatible HTTP calls (`call_gpt_segmented`).
  - `scripts/benchmark/annotation.py` — vLLM-based per-turn 9-field extractor; supports resumable JSONL output and `extract_first_json_array` fallback.
  - `scripts/benchmark/objective_eval.py` — concrete metric formulas (StrategyDensity, IKT, BP, etc.) and TotalScore weights.
  - `scripts/benchmark/subjective_eval.py` — judge-model prompts, JSON-parsing fallbacks, threaded evaluation and report generation.
  - `collabllm/reward.py` — multiturn-aware reward helpers used by training scripts.
  - `collabllm/prompts/` — canonical prompts used across scripts.

- Environment & run hints (must-check before running):

  - `.env` controls API endpoints and model names. Common keys: `OPENAI_API_KEY`, `OPENAI_CHAT_COMPLETIONS_URL` (or `OPENAI_BASE_URL`), `ANNOTATION_MODEL`, `ANNOTATION_TOKENIZER`, `JUDGE_API_URL`, `OUTPUT_DIR`.
  - Use a Python venv/conda matching `pyproject.toml` / `collabllm/requirements.txt`. README notes Python >= 3.13.

- Project conventions you must follow in code changes:

  - Outputs: generation → `multi_dialogue_topic_<topic_id>.json`; annotation → one JSON object per line in `.jsonl` with added `annotations` field.
  - Prompts and output formats are strict: `annotation.py` expects a JSON array per prompt; `subjective_eval.py` expects pure JSON from the judge model. If changing prompts, update parsing fallbacks too.
  - Resumable jobs: `annotation.py` and other batch scripts check existing outputs and resume—preserve file naming and line order when modifying behavior.

- Integration notes:

  - Generation and evaluation use OpenAI-compatible HTTP; calls use `requests` wrappers — do not assume `openai` SDK only.
  - Annotation uses local `vllm` (GPU recommended) + HuggingFace tokenizer paths; heavy GPU resources expected for annotation.
  - Judge model is an HTTP service (`JUDGE_API_URL`) - adjust `stop` tokens and temperature in prompts to keep responses JSON.

- Where to add data-cleaning utilities:

  - For reusable helpers (importable): `collabllm/datasets/cleaning.py` or `collabllm/utils/cleaning.py`.
  - For an executable pipeline step: `scripts/engine/clean_data.py` (matches existing `build_dataset.py` placement).

- Quick commands (examples):
  - Generate dialogues: `python scripts/benchmark/multi_dialogue.py` (ensure `.env` set)
  - Annotate: `python scripts/benchmark/annotation.py --input-dir datasets/raw --output-dir datasets/annotated`
  - Objective eval: `python scripts/benchmark/objective_eval.py datasets/annotated/*.jsonl --summary-file outputs/_overall_summary_metrics.json`

If anything here is unclear or you want deeper examples (prompt text, metric lines, or a cleaning script template), tell me which part to expand.
```

**Feedback:** Please review and suggest improvements or clarify any unclear sections. This guide will be iterated for completeness and accuracy.
