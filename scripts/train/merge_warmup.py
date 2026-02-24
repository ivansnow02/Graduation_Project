#!/usr/bin/env python3
"""
Merge a warmup LoRA model into base weights and save as a standalone merged model.

Example:
    uv run scripts/train/merge_warmup.py \
        --model_name outputs/sid_warmup_attention_only \
        --output_dir outputs/qwen14b_warmup_merged \
        --max_seq_length 4096 \
        --save_method merged_16bit
"""

from __future__ import annotations

import argparse
import os
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Merge warmup LoRA into base model")
    p.add_argument(
        "--model_name",
        type=str,
        default="outputs/sid_warmup_attention_only",
        help="Path or name of the warmup model (with LoRA adapters).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen14b_warmup_merged",
        help="Where to save merged standalone model.",
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument(
        "--save_method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit","merged_4bit_forced", "merged_4bit", "lora"],
        help="Unsloth save mode for merged export.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit before merging (not recommended for merged_16bit).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading warmup model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    print(
        f"Merging LoRA into base model -> {args.output_dir} "
        f"(save_method={args.save_method})"
    )
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method=args.save_method,
    )
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Merged model saved.")


if __name__ == "__main__":
    main()
