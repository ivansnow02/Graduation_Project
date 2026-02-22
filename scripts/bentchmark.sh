uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpo_500_3can/ -m qwen-flash -o data/batch/dpo_500_3can/batch.jsonl

uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpo_500_3can/ -b data/batchoutput/dpo500/batch.jsonl -o data/final/dpo_500_3can/

uv run scripts/benchmark/objective_eval.py data/final/dpo_500_3can/*.jsonl    --summary-file data/metrics/dpo500_3can.json    --output-dir data/metrics/dpo500_per_file_metrics

## baseline
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/qwen-14b/ -m qwen-flash -o data/batch/baseline/batch.jsonl

uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/qwen-14b/ -b data/batchoutput/dpo500/batch.jsonl -o data/final/baseline/

uv run scripts/benchmark/objective_eval.py data/final/baseline/*.jsonl    --summary-file data/metrics/baseline.json    --output-dir data/metrics/baseline_per_file_metrics

## sft
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/qwen-14b-sid-sft-2500/ -m qwen-flash -o data/batch/sft/batch.jsonl

uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/qwen-14b-sid-sft-2500/ -b data/batchoutput/sft/batch.jsonl -o data/final/sft/

uv run scripts/benchmark/objective_eval.py data/final/sft/*.jsonl    --summary-file data/metrics/sft.json    --output-dir data/metrics/sft_per_file_metrics
