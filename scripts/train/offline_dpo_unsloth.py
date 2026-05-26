

from __future__ import annotations

import argparse
import os
import sys

# 重要：用于 DPO 时必须先 patch Unsloth，再导入 transformers
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

    # 数据与路径
    p.add_argument("--dataset_repo", type=str, required=True, help="Path to dataset")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--min_score_gap", type=float, default=0.05)

    # 模型
    p.add_argument(
        "--model_name", type=str, required=True, default="Qwen/Qwen3-14B-Instruct"
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=True)

    # LoRA 配置
    p.add_argument("--peft_r", type=int, default=64)
    p.add_argument("--peft_alpha", type=int, default=32)
    p.add_argument("--peft_dropout", type=float, default=0)  # Unsloth 支持 0
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # 训练参数
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

    # 其他
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 使用 Unsloth 加载模型
    print(f"Loading Unsloth model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # 应用聊天模板，如有需要可针对 Qwen 特化，也可以依赖 tokenizer_config.json
    # Zephyr 示例依赖正确的模板应用
    # Qwen 2.5 通常已有合适模板，但必要时可以显式指定
    from unsloth.chat_templates import get_chat_template

    # 如果使用 Qwen 模型，确保模板正确
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # 应用 LoRA；如果加载的模型已经包含适配器，则跳过
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

    # 加载数据集
    print(f"Loading dataset from {args.dataset_repo}...")
    import json
    from datasets import Dataset, DatasetDict
    with open(args.dataset_repo, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if "chosen" in raw_data[0]:
        print("Detected pre-formatted DPO dataset.")
        import random
        full_ds = Dataset.from_list(raw_data)
        k = int(args.eval_ratio * len(full_ds))
        eval_idx = set(random.sample(range(len(full_ds)), k=k))
        train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]
        ds = DatasetDict({
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        })
    else:
        ds = MultiturnDataset(args.dataset_repo).to_dpo_dataset(
            eval_ratio=args.eval_ratio, minimum_gap=args.min_score_gap
        )

    # 格式化数据集
    # 如模板自动添加 assistant 前缀，这里负责去除
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    # 标准 Qwen assistant 起始标记
    assistant_prefix = "<|im_start|>assistant\n"

    def process(row):
        # Zephyr 示例对 prompt 使用 `apply_chat_template`
        # 这里假设 `row["prompt"]` 是消息列表（到 user 为止），`chosen` / `rejected` 是回复字符串

        # 为 prompt 应用模板
        # `add_generation_prompt=True` 可确保结尾带有 assistant 起始标记
        if isinstance(row["prompt"], list):
            row["prompt"] = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )

        # 格式化 chosen / rejected 回复
        # 注意：DPO 期望 chosen / rejected 只是回复文本，不是完整对话
        # 但如果使用聊天模板，仍需特别注意
        # 通常在 TRL 的 DPO 中，我们提供 prompt（历史）和 chosen / rejected（仅回复）

        # 确保结尾带有 EOS token
        if not row["chosen"].endswith(tokenizer.eos_token):
            row["chosen"] = row["chosen"] + tokenizer.eos_token
        if not row["rejected"].endswith(tokenizer.eos_token):
            row["rejected"] = row["rejected"] + tokenizer.eos_token

        if "margin" in row:
            row["margin"] = float(row["margin"])

        return row

    print("Formatting dataset...")
    ds = ds.map(process, num_proc=4, load_from_cache_file=False)

    # 配置 Trainer
    dpo_config_kwargs = dict(
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
        loss_type="sigmoid",
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

    try:
        training_args = DPOConfig(**dpo_config_kwargs)
    except TypeError as exc:
        print(
            "Current TRL does not accept loss_type='ipo'. Falling back to default DPO loss."
        )
        print(f"Compatibility detail: {exc}")
        dpo_config_kwargs.pop("loss_type", None)
        training_args = DPOConfig(**dpo_config_kwargs)

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

    # 训练
    print(f"Starting DPO training...")
    dpo_trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    # 保存
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Fix for GGUF saving if needed later
    # if args.save_gguf: ...


if __name__ == "__main__":
    main()
