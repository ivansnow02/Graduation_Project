"""
Unsloth-accelerated DPO training script.
Based on scripts/train/offline_dpo.py, scripts/train/sft_unsloth.py and zephyr_(7b)_dpo.py.
"""

from __future__ import annotations

import argparse
import os
import sys

# IMPORTANT: Unsloth must be patched before transformers for DPO
from unsloth import PatchDPOTrainer, FastLanguageModel, is_bfloat16_supported

PatchDPOTrainer()

import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from collabllm.datasets.multiturn import MultiturnDataset
import re

try:
    from swanlab.integration.transformers import SwanLabCallback

    SWANLAB_INSTALLED = True
except ImportError:
    SWANLAB_INSTALLED = False


def parse_args() -> argparse.Namespace:
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    p = argparse.ArgumentParser("Unsloth-accelerated Offline DPO Trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True, help="Path to dataset")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--min_score_gap", type=float, default=0.0)

    # Model
    p.add_argument(
        "--model_name", type=str, required=True, default="Qwen/Qwen3-14B-Instruct"
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=True)

    # LoRA config
    p.add_argument("--peft_r", type=int, default=64)
    p.add_argument("--peft_alpha", type=int, default=32)
    p.add_argument("--peft_dropout", type=float, default=0)  # Unsloth supports 0
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Training args
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument(
        "--save_only_model",
        action="store_true",
        help="Only save model weights (skip optimizer/scheduler state).",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit total number of checkpoints. Older ones are deleted.",
    )
    p.add_argument(
        "--load_best_model_at_end",
        type=_str2bool,
        default=False,
        help="Load best checkpoint at training end.",
    )
    p.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_rewards/margins",
        help="Metric name used to select best checkpoint.",
    )
    p.add_argument(
        "--greater_is_better",
        type=_str2bool,
        default=True,
        help="Whether larger metric value indicates better model.",
    )

    # Misc
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Model with Unsloth ---
    print(f"Loading Unsloth model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Apply Chat Template (Qwen specific if needed, or rely on tokenizer_config.json)
    # Zephyr example relies on proper template application.
    # Qwen 2.5 usually has good template, but we can enforce it if needed.
    from unsloth.chat_templates import get_chat_template

    # If using Qwen models, ensure template is correct
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # Apply LoRA (skip if the loaded model already contains adapters)
    has_existing_lora = hasattr(model, "peft_config") and bool(
        getattr(model, "peft_config", None)
    )
    if has_existing_lora:
        print(
            "Detected existing LoRA adapters in the loaded model. "
            "Skipping FastLanguageModel.get_peft_model(...)."
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.peft_r,
            target_modules=args.target_modules.split(","),
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    # --- 2. Load Dataset ---
    print(f"Loading dataset from {args.dataset_repo}...")
    ds = MultiturnDataset(args.dataset_repo).to_dpo_dataset(
        eval_ratio=args.eval_ratio, minimum_gap=args.min_score_gap
    )

    # --- 3. Format Dataset ---
    # Helper to strip assistant prefix if template adds it
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    # Standard Qwen assistant start token
    assistant_prefix = "<|im_start|>assistant\n"

    def process(row):
        # Zephyr example uses apply_chat_template for prompt
        # We assume row["prompt"] is messages list (upto user), row["chosen"]/["rejected"] are response strings

        # Apply template to prompt
        # add_generation_prompt=True ensures it ends with assistant start token
        if isinstance(row["prompt"], list):
            row["prompt"] = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )

        # Format chosen/rejected responses
        # NOTE: DPO expects chosen/rejected to be just the response text, NOT full convo
        # However, if using chat template, sometimes we need to be careful.
        # But typically for DPO with TRL, we provide prompt (history) and chosen/rejected (response only).

        # Ensure EOS token is at the end
        if not row["chosen"].endswith(tokenizer.eos_token):
            row["chosen"] = row["chosen"] + tokenizer.eos_token
        if not row["rejected"].endswith(tokenizer.eos_token):
            row["rejected"] = row["rejected"] + tokenizer.eos_token

        return row

    print("Formatting dataset...")
    ds = ds.map(process, num_proc=4, load_from_cache_file=False)

    # --- 4. Configure Trainer ---
    training_args = DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=args.output_dir,
        report_to="wandb" if args.wandb_project else "none",
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        gradient_checkpointing=True,
    )

    # Callbacks
    callbacks = []
    if args.use_swanlab and SWANLAB_INSTALLED:
        callbacks.append(
            SwanLabCallback(
                project=args.wandb_project or "sid-qwen3-14b-offline-dpo",
                run_name=os.path.basename(args.output_dir),
            )
        )

    # Initialize Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles ref_model internally
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # --- 5. Train ---
    print(f"Starting DPO training...")
    dpo_trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    # --- 6. Save ---
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Fix for GGUF saving if needed later
    # if args.save_gguf: ...


if __name__ == "__main__":
    main()
