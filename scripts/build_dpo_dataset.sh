source .venv/bin/activate
uv run scripts/engine/build_dataset.py \
    --dataset_name interdisciplinary \
    --metric_names "teaching_quality" \
    --metric_weights 1.0 \
    --num_candidate_responses 2 \
    --train_size 1 \
    --output_dir outputs/test_dpo \
    --user_generation_kwargs '{"model": "openai/qwen-plus", "require_json": false}' \
    --user_prompt_file "collabllm/prompts/student_simulator.txt" \
    --assistant_generation_kwargs '{"model": "openai/qwen-plus"}' \
    --reward_generation_kwargs '{"model": "openai/qwen-plus"}' \
    --proact_prompt_ratio 0 \
    --add_system_prompt_ratio 1 \
    --resume