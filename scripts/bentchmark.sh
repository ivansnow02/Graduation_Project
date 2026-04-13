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
mkdir -p data/batch/dpo_model_1k_3can_rewritten_opt
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpo_model_1k_3can_rewritten_opt -m qwen-flash -o data/batch/dpo_model_1k_3can_rewritten_opt/batch.jsonl


mkdir -p data/final/dpo_model_1k_3can_rewritten_opt data/batchoutput/dpo_model_1k_3can_rewritten_opt
uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpo_model_1k_3can_rewritten_opt/ -b data/batchoutput/dpo_model_1k_3can_rewritten_opt/dpo.jsonl -o data/final/dpo_model_1k_3can_rewritten_opt/


mkdir -p data/metrics/dpo_model_1k_3can_rewritten_opt_per_file_metrics
uv run scripts/benchmark/objective_eval.py data/final/dpo_model_1k_3can_rewritten_opt/*.jsonl    --summary-file data/metrics/dpo_model_1k_3can_rewritten_opt.json    --output-dir data/metrics/dpo_model_1k_3can_rewritten_opt_per_file_metrics


uv run scripts/data_prep/rewrite_test.py -n 10 -m openai/qwen-plus --annotation-model openai/qwen-flash

# data/dialog/dpobaseline_dialog_opt_2
mkdir -p data/batch/dpobaseline_dialog_opt_2
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpobaseline_dialog_opt_2 -m qwen-flash -o data/batch/dpobaseline_dialog_opt_2/batch.jsonl


mkdir -p data/final/dpobaseline_dialog_opt_2 data/batchoutput/dpobaseline_dialog_opt_2
uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpobaseline_dialog_opt_2/ -b data/batchoutput/dpobaseline_dialog_opt_2/*.jsonl -o data/final/dpobaseline_dialog_opt_2/


mkdir -p data/metrics/dpobaseline_dialog_opt_2_per_file_metrics
uv run scripts/benchmark/objective_eval.py data/final/dpobaseline_dialog_opt_2/*.jsonl    --summary-file data/metrics/dpobaseline_dialog_opt_2.json    --output-dir data/metrics/dpobaseline_dialog_opt_2_per_file_metrics


## subjective_eval for dpobaseline_dialog_opt_2
mkdir -p data/batch/dpobaseline_dialog_opt_2_subjective
uv run scripts/benchmark/subjective_eval_batch.py prepare -i data/dialog/dpobaseline_dialog_opt_2 -o data/batch/dpobaseline_dialog_opt_2_subjective/batch_requests.jsonl


mkdir -p data/final/dpobaseline_dialog_opt_2_subjective data/batchoutput/dpobaseline_dialog_opt_2_subjective
uv run scripts/benchmark/subjective_eval_batch.py merge -i data/dialog/dpobaseline_dialog_opt_2 -b data/batchoutput/dpobaseline_dialog_opt_2_subjective -o data/final/dpobaseline_dialog_opt_2_subjective

## dpo soft
mkdir -p data/batch/dpo_model_500_soft
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpo_model_500_soft -m qwen-flash -o data/batch/dpo_model_500_soft/batch.jsonl


mkdir -p data/final/dpo_model_500_soft data/batchoutput/dpo_model_500_soft
uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpo_model_500_soft/ -b data/batchoutput/dpo_model_500_soft/*.jsonl -o data/final/dpo_model_500_soft/


mkdir -p data/metrics/dpo_model_500_soft_per_file_metrics
uv run scripts/benchmark/objective_eval.py data/final/dpo_model_500_soft/*.jsonl    --summary-file data/metrics/dpo_model_500_soft.json    --output-dir data/metrics/dpo_model_500_soft_per_file_metrics


# dpobaseline_soft
mkdir -p data/batch/dpobaseline_soft
uv run scripts/benchmark/annotation_batch.py prepare -i data/dialog/dpobaseline_soft -m qwen-flash -o data/batch/dpobaseline_soft/batch.jsonl


mkdir -p data/final/dpobaseline_soft data/batchoutput/dpobaseline_soft
uv run scripts/benchmark/annotation_batch.py merge -i data/dialog/dpobaseline_soft/ -b data/batchoutput/dpobaseline_soft/*.jsonl -o data/final/dpobaseline_soft/


mkdir -p data/metrics/dpobaseline_soft_per_file_metrics
uv run scripts/benchmark/objective_eval.py data/final/dpobaseline_soft/*.jsonl    --summary-file data/metrics/dpobaseline_soft.json    --output-dir data/metrics/dpobaseline_soft_per_file_metrics