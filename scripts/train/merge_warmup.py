#!/usr/bin/env python3
"""
将 warmup 阶段的 LoRA 模型合并到基座权重，并保存为独立的合并模型。

示例：
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
    p = argparse.ArgumentParser("将 warmup LoRA 合并到基座模型")
    p.add_argument(
        "--model_name",
        type=str,
        default="outputs/sid_warmup_attention_only",
        help="warmup 模型的路径或名称，需包含 LoRA 适配器。",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen14b_warmup_merged",
        help="合并后独立模型的保存位置。",
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument(
        "--save_method",
        type=str,
        default="merged_4bit_forced",
        choices=["merged_16bit", "merged_4bit_forced", "merged_4bit", "lora"],
        help="Unsloth 的合并导出保存模式。",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="合并前以 4bit 加载模型，不建议与 `merged_16bit` 同时使用。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"正在加载 warmup 模型：{args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    quantization_config = getattr(model.config, "quantization_config", None)
    if quantization_config is not None and not isinstance(quantization_config, dict):
        try:
            model.config.quantization_config = quantization_config.to_dict()
        except Exception:
            model.config.quantization_config = None

    print(
        f"正在将 LoRA 合并到基座模型 -> {args.output_dir} "
        f"（save_method={args.save_method}）"
    )
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method=args.save_method,
    )
    tokenizer.save_pretrained(args.output_dir)

    config_path = os.path.join(args.output_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            "合并导出未生成 `config.json`。"
            "请改用 `--save_method merged_4bit_forced` 或 `merged_4bit` 重试。"
        )

    print("完成，合并模型已保存。")


if __name__ == "__main__":
    main()
