#!/usr/bin/env python3
"""Simple inference script for Unsloth-finetuned model saved in local directory.
Usage:
  python3 scripts/infer.py --model_dir outputs/sid_unsloth_sft --load_in_4bit
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


# ensure collabllm utilities on path
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":/workspace/collabllm"

try:
    # Import unsloth after setting PYTHONPATH/offline env if needed
    from unsloth import FastLanguageModel
except Exception:
    raise


def parse_args():
    p = argparse.ArgumentParser("infer")
    p.add_argument("--model_dir", type=str, default="outputs/dpo_model_1k_3can_opt")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. If omitted, uses default Chinese question.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Set HF/Transformers to offline mode before loading.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Loading model from {args.model_dir} (4bit={args.load_in_4bit})...")
    # try local_files_only first
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_dir,
            load_in_4bit=args.load_in_4bit,
            local_files_only=True,
        )
    except TypeError:
        # some FastLanguageModel wrappers may not accept local_files_only
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_dir,
            load_in_4bit=args.load_in_4bit,
        )

    # default prompt
    if args.prompt:
        user_text = args.prompt
    else:
        user_text = "秦朝怎么建立的？"


    system_prompt = """你是一位跨学科的教师，始终使用苏格拉底式提问法来引导学生。你的目标不是直接给出答案，而是通过逐轮递进、循序渐进的问题，引导学生独立思考并构建跨学科理解。
你的行为规范如下：
1. 每一轮只能提出一个简洁的问题；必须体现认知推进；不能重复提问；禁止多个并列问题。
2. 你应从学生上一次回答中提炼关键点，沿着一个核心问题主线继续深入。
3. 你应始终鼓励学生跨学科思考（生物、地理、物理、历史等）。
4. 当你判断学生理解已经完成，请生成一句简洁明了的总结，明确指出学生已经完成推理，并标注[结束]。
"""
    convo = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    # apply chat template if available
    try:
        prompt = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True,enable_thinking = False
        )
    except Exception:
        # fallback to raw user_text
        prompt = user_text

    print("Prompt:\n", prompt)

    # tokenize to tensors
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    except TypeError:
        # some tokenizer interfaces require encode + build inputs
        ids = tokenizer.encode(prompt)
        inputs = {"input_ids": torch.tensor([ids])}

    # move to model device
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # generate
    print("Generating...")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.95)

    # decode
    text = None
    try:
        if isinstance(out, torch.Tensor):
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        elif isinstance(out, (list, tuple)):
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            text = str(out)
    except Exception:
        # last resort: str()
        text = str(out)

    print("\n=== Output ===\n")
    print(text)


if __name__ == "__main__":
    main()
