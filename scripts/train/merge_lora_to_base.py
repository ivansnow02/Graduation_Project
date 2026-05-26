#!/usr/bin/env python3
"""
将 LoRA 适配器合并到原始 fp16 基座模型，并保存为干净的 16 位模型。
该脚本在 CPU 上运行以避免显存限制，14B 模型通常需要约 32GB 系统内存。
基座模型从 ModelScope 下载。

用法：
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
    p = argparse.ArgumentParser("将 LoRA 适配器合并到 fp16 基座模型")
    p.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="ModelScope 模型 ID，例如 Qwen/Qwen3-14B。",
    )
    p.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/dpo500softmargin",
        help="LoRA 适配器目录，需包含 `adapter_model.safetensors` 和 `adapter_config.json`。",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dpo500softmargin_merged_16bit",
        help="合并后 16 位模型的保存位置。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"正在从 ModelScope 下载基座模型：{args.base_model}")
    model_dir = snapshot_download(
        args.base_model,
        cache_dir=CACHE_DIR,
    )
    print(f"基座模型缓存位置：{model_dir}")

    print("正在以 CPU 加载基座模型，这可能需要几分钟……")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"正在加载 LoRA 适配器：{args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("正在将 LoRA 合并到基座模型……")
    model = model.merge_and_unload()

    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    print(f"正在保存合并模型到：{args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)

    adapter_template = os.path.join(args.adapter_dir, "chat_template.jinja")
    if os.path.exists(adapter_template):
        shutil.copy2(adapter_template, os.path.join(args.output_dir, "chat_template.jinja"))

    import json
    config_path = os.path.join(args.output_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError("合并失败：未生成 `config.json`。")
    with open(config_path) as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print("完成，16 位合并模型已保存。")


if __name__ == "__main__":
    main()
