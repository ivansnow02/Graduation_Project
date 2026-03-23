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
STUDENT_LOCAL_MODEL="unsloth/Qwen3-14B-unsloth-bnb-4bit" \
TEACHER_BASE_MODEL="outputs/qwen14b_warmup_merged_4bit" \
TEACHER_ADAPTER_MODEL="outputs/dpo_model_1k_3can_opt" \
MULTI_DIALOGUE_WORKERS=1 \
uv run scripts/benchmark/multi_dialogue_unsloth.py 2>&1 | tee outputs/logs/multi_dialogue_unsloth.log ; /usr/bin/shutdown


uv run scripts/train/offline_dpo_unsloth.py \
    --dataset_repo "outputs/dpo_mix_1000_3can/interdisciplinary_multiturn.json" \
    --output_dir "outputs/dpo_mix_1000_3can_opt" \
    --model_name "outputs/qwen14b_warmup_merged" \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --min_score_gap 0.05 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_rewards/margins" \
    --greater_is_better True \
    --save_only_model \
    --use_swanlab; /usr/bin/shutdown


## sft
uv run scripts/train/sft_unsloth.py \
    --model_name unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --dataset_repo data/converted_2500.jsonl \
    --output_dir outputs/sid_warmup_attention_only \
    --max_seq_length 4096 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --load_in_4bit \
    --peft_r 16 \
    --peft_alpha 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --warmup_steps 10 \
    --use_swanlab; shutdown

# merge warmup LoRA into a standalone base model
uv run scripts/train/merge_warmup.py \
    --model_name outputs/sid_warmup_attention_only \
    --output_dir outputs/qwen14b_warmup_merged_4bit \
    --max_seq_length 4096 \
    --load_in_4bit \
    --save_method merged_4bit_forced 2>&1 | tee outputs/logs/merge_warmup.log; shutdown


uv run scripts/train/offline_dpo_unsloth.py \
    --dataset_repo "outputs/dpo_mix_1000_3can/interdisciplinary_multiturn.json" \
    --output_dir "outputs/dpo_model_1k_3can_opt" \
    --model_name "outputs/qwen14b_warmup_merged_4bit" \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --peft_r 64 \
    --peft_alpha 128 \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_rewards/margins" \
    --greater_is_better True \
    --save_only_model \
    --use_swanlab; shutdown

uv run scripts/train/merge_warmup.py \
    --model_name outputs/dpo_model_1k_3can_opt \
    --output_dir /root/autodl-fs/output/qwen14b_socratic_16bit \
    --max_seq_length 4096 \
    --load_in_4bit \
    --save_method merged_16bit 2>&1 | tee outputs/logs/merge_warmup.log; shutdown
vllm serve unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules teacher_model=outputs/dpo_model_500_3can_opt \
    --port 8000


uv run scripts/benchmark/multi_dialogue.py ; /usr/bin/shutdown


uv run --project . scripts/train/online_dpo_unsloth.py \
    --dataset_name interdisciplinary \
    --dataset_repo "outputs/dpo_500_3can_2/interdisciplinary_multiturn.json" \
    --output_dir "outputs/online_dpo_model_500_3can" \
    --model_name "unsloth/Qwen3-14B-unsloth-bnb-4bit" \
    --metric_names "teaching_quality" \
    --metric_weights 1.0 \
    --user_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "require_json": false, "temperature": 1.0, "max_tokens": 2048}' \
    --assistant_generation_kwargs '{"model": "openai/unsloth/Qwen3-14B-unsloth-bnb-4bit", "base_url": "http://0.0.0.0:8000/v1", "api_key": "not-needed", "temperature": 1.0, "max_tokens": 2048}' \
    --reward_generation_kwargs '{"model": "openai/qwen-plus"}' \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 10 \
    --num_samples 3 \
    --max_new_turns 4 \
    --max_metric_workers 2 \
    --save_only_model \
    --save_total_limit 3 \
    --use_vllm \
    --use_swanlab; /usr/bin/shutdown

uv run scripts/train/sft_unsloth.py \
    --model_name unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --dataset_repo data/sft_mixed.jsonl \
    --output_dir outputs/sid_cqia_mixed_r64 \
    --max_seq_length 4096 \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --load_in_4bit \
    --peft_r 64 \
    --peft_alpha 128 \
    --learning_rate 1.5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --packing \
    --use_swanlab; shutdown

uv run scripts/data_prep/rewrite_batch.py merge \
    -i data/rewrite_batch/dpo_pairs.json \
    -b data/rewrite_batch/batch_output.jsonl \
    -o data/rewrite_batch/dpo_pairs_rewritten.json
