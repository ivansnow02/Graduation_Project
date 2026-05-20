#!/usr/bin/env python3
"""
Merge a LoRA adapter into the original fp16 base model, saving as clean 16-bit.
Works on CPU to avoid VRAM limits — needs ~32GB system RAM for 14B models.
Downloads base model from ModelScope.

Usage:
    uv run python scripts/train/merge_lora_to_base.py \
        --base_model Qwen/Qwen3-14B \
        --adapter_dir outputs/dpo500softmargin \
        --output_dir outputs/dpo500softmargin_merged_16bit
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download


CACHE_DIR = "/home/ioyuk1nya/Graduation_Project/.cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Merge LoRA adapter into fp16 base model")
    p.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="ModelScope model ID (e.g. Qwen/Qwen3-14B).",
    )
    p.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/dpo500softmargin",
        help="Directory with LoRA adapter (adapter_model.safetensors + adapter_config.json).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dpo500softmargin_merged_16bit",
        help="Where to save the merged 16-bit model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading base model from ModelScope: {args.base_model}")
    model_dir = snapshot_download(
        args.base_model,
        cache_dir=CACHE_DIR,
    )
    print(f"Base model cached at: {model_dir}")

    print("Loading base model on CPU (this may take a few minutes)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)

    adapter_template = os.path.join(args.adapter_dir, "chat_template.jinja")
    if os.path.exists(adapter_template):
        shutil.copy2(adapter_template, os.path.join(args.output_dir, "chat_template.jinja"))

    import json
    config_path = os.path.join(args.output_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError("Merge failed: config.json not produced.")
    with open(config_path) as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print("Done. Merged 16-bit model saved.")


if __name__ == "__main__":
    main()
