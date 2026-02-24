mkdir -p data/batch/dpo_1k_3can
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpo_1k_3can -m qwen-flash -o data/batch/dpo_1k_3can/batch.jsonl


mkdir -p data/final/dpo_1k_3can
uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpo_1k_3can/ -b data/batchoutput/dpo1k/dpo.jsonl -o data/final/dpo_1k_3can/


mkdir -p data/metrics/dpo1k_per_file_metrics
uv run scripts/benchmark/objective_eval.py data/final/dpo_1k_3can/*.jsonl    --summary-file data/metrics/dpo1k_3can.json    --output-dir data/metrics/dpo1k_per_file_metrics

## baseline
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/qwen-14b/ -m qwen-flash -o data/batch/baseline/base.jsonl

uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/qwen-14b/ -b data/batchoutput/base/base.jsonl -o data/final/baseline/

uv run scripts/benchmark/objective_eval.py data/final/baseline/*.jsonl    --summary-file data/metrics/baseline.json    --output-dir data/metrics/baseline_per_file_metrics

## sft
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/qwen-14b-sid-sft-2500/ -m qwen-flash -o data/batch/sft/batch.jsonl

uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/qwen-14b-sid-sft-2500/ -b data/batchoutput/sft/sft.jsonl -o data/final/sft/

uv run scripts/benchmark/objective_eval.py data/final/sft/*.jsonl    --summary-file data/metrics/sft.json    --output-dir data/metrics/sft_per_file_metrics
