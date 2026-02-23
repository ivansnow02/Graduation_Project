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

vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --port 8000
# 使用 tee 同时输出到终端和日志文件
uv run --project . scripts/engine/build_dataset.py \
    --dataset_name interdisciplinary \
    --metric_names "teaching_quality" \
    --metric_weights 1.0 \
    --num_candidate_responses 3 \
    --train_size 500 \
    --output_dir outputs/dpo_base_500_3can \
    --user_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 2048}' \
    --user_prompt_file "collabllm/prompts/student_simulator.txt" \
    --assistant_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 2048}' \
    --reward_generation_kwargs '{"model": "openai/qwen-flash",  "temperature": 0}' \
    --proact_prompt_ratio 0 \
    --add_system_prompt_ratio 1 \
    --resume \
    --allow_repeat_samples 2>&1 | tee outputs/logs/dpo_base_500_3can_build.log ; /usr/bin/shutdown

#dpo training offline
uv run scripts/train/offline_dpo_unsloth.py \
    --dataset_repo "outputs/dpo_500_3can/interdisciplinary_multiturn.json" \
    --output_dir "outputs/dpo_model_500_3can_opt" \
    --model_name "outputs/sid_qwen14b_sft_2500" \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --eval_steps 10 \
    --save_steps 10 \
    --min_score_gap 0.05 \
    --save_only_model \
    --save_total_limit 3 \
    --resume_ckpt_dir "outputs/dpo_model_500_3can_opt/checkpoint-80" \
    --use_swanlab; /usr/bin/shutdown

uv run --project . scripts/engine/build_dataset.py     --dataset_name interdisciplinary     --metric_names "teaching_quality"     --metric_weights 1.0     --num_candidate_responses 2     --train_size 5     --output_dir outputs/dpo_5     --user_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 2048}'     --user_prompt_file "collabllm/prompts/student_simulator.txt"     --assistant_generation_kwargs '{"model": "openai/teacher_model", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "temperature": 1.0, "max_tokens": 2048}'     --reward_generation_kwargs '{"model": "openai/qwen-plus"}'     --proact_prompt_ratio 0    --add_system_prompt_ratio 1    --resume    --allow_repeat_samples

uv run scripts/data_prep/to_dpo_format.py --input_file outputs/dpo_5/interdisciplinary_multiturn.json --output_dir outputs/dpo_formatted_5


#dpo bench
vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules teacher_model=outputs/dpo_model_500_3can_opt \
    --port 8000


uv run scripts/benchmark/multi_dialogue.py ; /usr/bin/shutdown


uv run scripts/train/offline_dpo_unsloth.py \
    --dataset_repo "outputs/dpo_base_500_3can/interdisciplinary_multiturn.json" \
    --output_dir "outputs/dpo_base_500_3can_opt" \
    --model_name "unsloth/Qwen3-14B-unsloth-bnb-4bit" \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --eval_steps 10 \
    --save_steps 10 \
    --min_score_gap 0.05 \
    --save_only_model \
    --save_total_limit 3 \
    --use_swanlab; /usr/bin/shutdown


## sft

