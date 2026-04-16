#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/build_dpo_dataset.sh <build|train|eval|all> <experiment_name> [judge_model]

Examples:
  ./scripts/build_dpo_dataset.sh build dpo_500_3can_update
  ./scripts/build_dpo_dataset.sh train dpo_500_3can_update
  ./scripts/build_dpo_dataset.sh eval dpo_500_3can_update qwen-flash
  ./scripts/build_dpo_dataset.sh all dpo_500_3can_update qwen-flash

Environment overrides:
  BASE_MODEL              default: openai/.cache/huggingface/hub/Qwen3-14B-unsloth-bnb-4bit
  BASE_URL                default: http://localhost:8000/v1
  REWARD_MODEL            default: openai/qwen-flash
  MODEL_NAME              default: outputs/sid_qwen14b_sft_2500
  TRAIN_SIZE              default: 500
  NUM_CANDIDATES          default: 3
  LR                      default: 2e-6
  BATCH_SIZE              default: 2
  GRAD_ACC                default: 16
  EPOCHS                  default: 3
  EVAL_STEPS              default: 10
  SAVE_STEPS              default: 10
  SAVE_TOTAL_LIMIT        default: 2
  RESUME_CKPT_DIR         default: (empty)
EOF
}

ACTION="${1:-}"
EXPERIMENT="${2:-}"
JUDGE_MODEL="${3:-qwen-flash}"

if [[ -z "${ACTION}" || -z "${EXPERIMENT}" ]]; then
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

LOG_DIR="outputs/logs"
mkdir -p "${LOG_DIR}"

DATASET_NAME="interdisciplinary"
BASE_MODEL="${BASE_MODEL:-openai/.cache/huggingface/hub/Qwen3-14B-unsloth-bnb-4bit}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
REWARD_MODEL="${REWARD_MODEL:-openai/qwen-flash}"
MODEL_NAME="${MODEL_NAME:-outputs/sid_qwen14b_sft_2500}"
TRAIN_SIZE="${TRAIN_SIZE:-500}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
LR="${LR:-2e-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
EPOCHS="${EPOCHS:-3}"
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RESUME_CKPT_DIR="${RESUME_CKPT_DIR:-}"

DATASET_OUTPUT_DIR="outputs/${EXPERIMENT}"
DATASET_PATH="${DATASET_OUTPUT_DIR}/${DATASET_NAME}_multiturn.json"
TRAIN_OUTPUT_DIR="outputs/${EXPERIMENT}_opt"

build_dataset() {
  echo "[build] experiment=${EXPERIMENT}"
  uv run scripts/engine/build_dataset.py \
    --dataset_name "${DATASET_NAME}" \
    --metric_names "teaching_quality" \
    --metric_weights 1.0 \
    --num_candidate_responses "${NUM_CANDIDATES}" \
    --train_size "${TRAIN_SIZE}" \
    --output_dir "${DATASET_OUTPUT_DIR}" \
    --user_generation_kwargs "{\"model\": \"${BASE_MODEL}\", \"base_url\": \"${BASE_URL}\", \"api_key\": \"not-needed\", \"require_json\": false, \"temperature\": 1.0, \"max_tokens\": 2048}" \
    --user_prompt_file "collabllm/prompts/student_simulator.txt" \
    --assistant_generation_kwargs "{\"model\": \"${BASE_MODEL}\", \"base_url\": \"${BASE_URL}\", \"api_key\": \"not-needed\", \"require_json\": false, \"temperature\": 1.0, \"max_tokens\": 2048}" \
    --reward_generation_kwargs "{\"model\": \"${REWARD_MODEL}\", \"temperature\": 0}" \
    --proact_prompt_ratio 0 \
    --add_system_prompt_ratio 1 \
    --resume \
    --allow_repeat_samples 2>&1 | tee "${LOG_DIR}/${EXPERIMENT}_build.log"
}

train_dpo() {
  echo "[train] dataset=${DATASET_PATH} output=${TRAIN_OUTPUT_DIR}"
  local resume_args=()
  if [[ -n "${RESUME_CKPT_DIR}" ]]; then
    resume_args=(--resume_ckpt_dir "${RESUME_CKPT_DIR}")
  fi

  uv run scripts/train/offline_dpo_unsloth.py \
    --dataset_repo "${DATASET_PATH}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --max_seq_length 4096 \
    --num_train_epochs "${EPOCHS}" \
    --logging_steps 1 \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_rewards/margins" \
    --greater_is_better True \
    --save_only_model \
    --use_swanlab \
    "${resume_args[@]}" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT}_train.log"
}

run_eval() {
  echo "[eval] experiment=${EXPERIMENT} judge_model=${JUDGE_MODEL}"
  ./scripts/run_benchmark.sh prepare "${EXPERIMENT}" "${JUDGE_MODEL}"
  ./scripts/run_benchmark.sh eval "${EXPERIMENT}"
}

case "${ACTION}" in
  build)
    build_dataset
    ;;
  train)
    train_dpo
    ;;
  eval)
    run_eval
    ;;
  all)
    build_dataset
    train_dpo
    run_eval
    ;;
  *)
    usage
    exit 1
    ;;
esac
