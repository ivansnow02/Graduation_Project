#!/usr/bin/env python3
"""
Modified SFT trainer using Unsloth for acceleration.
Optimized for Qwen 3 (and Qwen 2.5) with SID dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

# IMPORTANT: Unsloth must be imported before transformers
from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from unsloth.chat_templates import get_chat_template
import torch
from collabllm.datasets.multiturn import MultiturnDataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer



try:
    from swanlab.integration.transformers import SwanLabCallback
    SWANLAB_INSTALLED = True
except ImportError:
    SWANLAB_INSTALLED = False

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Unsloth accelerated SFT trainer for Qwen 3")
    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True, help="JSON file path")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.1)

    # Model - 默认改为 Qwen 3
    # 如果你是 4bit 版本，可以用 unsloth/Qwen3-14B-Instruct-bnb-4bit (如果 Unsloth 已发布)
    # 或者直接用原始权重配合 --load_in_4bit
    p.add_argument("--model_name", type=str, required=True, default="Qwen/Qwen3-14B-Instruct")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=True)

    # LoRA config
    p.add_argument("--peft_r", type=int, default=16)
    p.add_argument("--peft_alpha", type=int, default=16)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Training args
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=1) # SID 切片数据，跑 1 epoch 足矣
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=10)

    # Misc
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")
    p.add_argument("--save_gguf", action="store_true", help="Save model in GGUF format")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    # --- 1. 配置 Logger ---
    callbacks = []
    if args.use_swanlab and SWANLAB_INSTALLED:
        swanlab_callback = SwanLabCallback(
            project="sid-qwen3-14b-sft",
            run_name=args.output_dir.split("/")[-1]
        )
        callbacks.append(swanlab_callback)


    # --- 2. 加载数据 ---
    print(f"Loading dataset from {args.dataset_repo}...")
    ds = MultiturnDataset(args.dataset_repo).to_sft_dataset(eval_ratio=args.eval_ratio)

    # --- 3. 加载 Unsloth 模型 ---
    print(f"Loading Unsloth model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # --- 关键适配：Qwen 3 依然使用 ChatML 格式 ---
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.peft_r,
        target_modules=args.target_modules.split(","),
        lora_alpha=args.peft_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # --- 4. 数据格式化 ---
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    print("Formatting dataset...")
    ds = ds.map(formatting_prompts_func, batched=True)

    # --- 5. 配置 Collator (针对 Qwen 3) ---
    # Qwen 3 的标准回答起始符依然是 <|im_start|>assistant\n
    response_template = "<|im_start|>assistant\n"


    # --- 6. 配置 Trainer ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        # max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        args=training_args,
        callbacks=callbacks,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",  # Qwen 3 / 2.5 的标准回答头，非常准确
    )

    # --- 7. 开始训练 ---
    print(f"Starting training for Qwen 3 model: {args.model_name}")
    trainer_stats = trainer.train()

    # --- 8. 保存 ---
    print(f"Saving to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_gguf:
        print("Converting to GGUF...")
        try:
            model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method="q4_k_m")
        except Exception as e:
            print(f"GGUF saving failed: {e}")

if __name__ == "__main__":
    main()

# 确保在项目根目录下
