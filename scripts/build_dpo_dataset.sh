source .venv/bin/activate
uv run --project . scripts/engine/build_dataset.py \
    --dataset_name interdisciplinary \
    --metric_names "teaching_quality" \
    --metric_weights 1.0 \
    --num_candidate_responses 2 \
    --train_size 1 \
    --output_dir outputs/test_dpo \
    --user_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 1024}' \
    --user_prompt_file "collabllm/prompts/student_simulator.txt" \
    --assistant_generation_kwargs '{"model": "openai/teacher_model", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "temperature": 0.8, "max_tokens": 2048}' \
    --reward_generation_kwargs '{"model": "openai/qwen-plus"}' \
    --proact_prompt_ratio 0 \
    --add_system_prompt_ratio 1 \
    --resume

vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules teacher_model=outputs/sid_qwen14b_sft_2500 \
    --port 8000

uv run --project . scripts/engine/build_dataset.py     --dataset_name interdisciplinary     --metric_names "teaching_quality"     --metric_weights 1.0     --num_candidate_responses 2     --train_size 5     --output_dir outputs/dpo_5     --user_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 1024}'     --user_prompt_file "collabllm/prompts/student_simulator.txt"     --assistant_generation_kwargs '{"model": "openai/teacher_model", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "temperature": 0.9, "max_tokens": 2048}'     --reward_generation_kwargs '{"model": "openai/qwen-plus"}'     --proact_prompt_ratio 0    --add_system_prompt_ratio 1     --resume
