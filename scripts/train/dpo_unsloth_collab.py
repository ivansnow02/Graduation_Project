#!/usr/bin/env python3
"""Offline DPO training using Unsloth + collabllm's MultiturnDataset."""

import argparse
import os
import sys

import torch
from datasets import DatasetDict
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from swanlab.integration.transformers import SwanLabCallback

# Patch must happen before transformers import/usage in Unsloth
PatchDPOTrainer()

# Make sure local collabllm package is visible
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from collabllm.datasets.multiturn import MultiturnDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_path",
        default="/workspace/datasets/sid_collabllm/sid_dpo_nested.json",
    )
    p.add_argument("--model_name", default="/workspace/outputs/sid_unsloth_sft")
    p.add_argument("--output_dir", default="/workspace/outputs/sid_unsloth_dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--minimum_gap", type=float, default=0.0)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--max_prompt_length", type=int, default=3584)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    return p.parse_args()


def format_for_dpo(tokenizer, ds: DatasetDict):
    def _map(row):
        row["prompt"] = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        row["chosen"] = row["chosen"].strip() + tokenizer.eos_token
        row["rejected"] = row["rejected"].strip() + tokenizer.eos_token
        return row

    ds["train"] = ds["train"].map(_map, load_from_cache_file=False)
    ds["eval"] = ds["eval"].map(_map, load_from_cache_file=False)
    return ds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    swanlab_callback = SwanLabCallback(
        project="sid-qwen3-offline-dpo", run_name=args.output_dir.split("/")[-1]
    )
    print(f"Loading dataset from {args.dataset_path} ...")
    multiturn = MultiturnDataset(args.dataset_path)
    ds = multiturn.to_dpo_dataset(
        eval_ratio=args.eval_ratio, minimum_gap=args.minimum_gap
    )

    print(f"Loading SFT model from {args.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    ds = format_for_dpo(tokenizer, ds)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="epoch",
        push_to_hub=False,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        args=training_args,
        callbacks=[swanlab_callback],
    )

    print("Starting DPO training ...")
    trainer.train()

    print("Saving model ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
