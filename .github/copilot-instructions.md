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

- **End-to-End Pipeline**:
  1. Prepare topics (`datasets/interdisciplinary_topic.json`).
  2. Generate dialogues: `python multi_dialogue.py` (requires OpenAI API key).
  3. Annotate: `python annotation.py` (requires vLLM, GPU, tokenizer/model paths).
  4. Objective metrics: `python objective_eval.py datasets/annotated/*.jsonl --summary-file outputs/_overall_summary_metrics.json`.
  5. Subjective metrics: `python subjective_eval.py` (judge model API, outputs JSON reports).
- **Parallelization**: Data generation and annotation use multiprocessing/threading for speed.
- **Resumable Processing**: Annotation and evaluation scripts support checkpointing and resume from partial outputs.

## Project-Specific Patterns

- **Socratic Dialogue**: Teacher only asks questions, no direct answers. Student responses vary by sampled persona/scenario.
- **Annotation Format**: Each turn is annotated with 9 fixed fields (see `annotation.py` and README for details).
- **Metrics Calculation**: Objective metrics use strict formulas (see `objective_eval.py`), subjective metrics use judge model with JSON output.
- **File Naming**: Outputs follow patterns like `multi_dialogue_topic_<topic_id>.json`, annotated files as `.jsonl`.
- **Config via .env**: All scripts read config from environment variables. Always check `.env` for required keys.

## Integration Points

- **OpenAI API**: Used for dialogue generation and subjective evaluation. Configure endpoints in `.env`.
- **vLLM**: Local annotation requires vLLM and GPU. Set model/tokenizer paths in `.env`.
- **Judge Model API**: Subjective evaluation via HTTP API, model name in `.env`.

## Examples

- To add a new dataset: Place in `examples/single_turn_ds/`, register in `__init__.py`, and follow notebook tutorials.
- To add a new metric: Implement in `examples/metrics/`, register in `__init__.py`.

## Troubleshooting

- If annotation fails to extract JSON, lower model temperature or strengthen prompt constraints.
- If judge model returns non-JSON, check API output and adjust prompt/stop tokens.
- For IKT metric calculation, see README for numerator/denominator details.

## References

- See `collabllm/README.md` and `scripts/benchmark/README.md` for detailed workflow and format specs.
- `.env` template is required for all scripts—fill in real values before running.

---

**Feedback:** Please review and suggest improvements or clarify any unclear sections. This guide will be iterated for completeness and accuracy.
