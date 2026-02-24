#!/usr/bin/env python3
"""
Unsloth-accelerated Online DPO training script.

Combines:
  - Unsloth FastLanguageModel (4-bit + LoRA) for efficient model loading
  - TRL OnlineDPOTrainer for online preference learning
  - collabllm MultiturnRewardJudge for multi-turn reward evaluation

Unlike online_dpo.py (which uses torchrun + DeepSpeed + vLLM external launcher),
this script runs on a **single GPU** with Unsloth's memory optimisations and
optionally uses TRL's built-in vLLM support (`--use_vllm`).

Example
-------
ENABLE_COLLABLLM_LOGGING=0 uv run python -m scripts.train.online_dpo_unsloth \
    --dataset_name math-hard \
    --metric_names "accuracy" "interactivity" "token_amount" \
    --metric_weights 1 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-math-hard-qwen3"}' \
    --reward_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft_unsloth/collabllm-multiturn-math-hard \
    --output_dir outputs/online_dpo_unsloth/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --num_samples 3 \
    --max_new_turns 4 \
    --max_metric_workers 2 \
    --use_vllm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

# ──────────────────────────────────────────────────────────────
# IMPORTANT: Unsloth must be patched BEFORE transformers / TRL
# ──────────────────────────────────────────────────────────────
from unsloth import PatchDPOTrainer, FastLanguageModel, is_bfloat16_supported

PatchDPOTrainer()

import numpy as np
import torch
from dotenv import load_dotenv
from trl import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.judges import BasePairwiseJudge

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info
from examples.metrics import *  # noqa: F401, F403 — registers metric classes

try:
    from swanlab.integration.transformers import SwanLabCallback

    SWANLAB_INSTALLED = True
except ImportError:
    SWANLAB_INSTALLED = False

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Unsloth-accelerated Online DPO Trainer")

    # Data / task
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--max_new_turns", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_metric_workers", type=int, default=4)

    # Paths
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Model (Unsloth)
    p.add_argument(
        "--model_name", type=str, required=True, default="Qwen/Qwen3-14B-Instruct"
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true", default=True)

    # LoRA
    p.add_argument("--peft_r", type=int, default=64)
    p.add_argument("--peft_alpha", type=int, default=32)
    p.add_argument("--peft_dropout", type=float, default=0)  # Unsloth optimised at 0
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Training
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument(
        "--save_only_model",
        action="store_true",
        help="Only save model weights (skip optimizer / scheduler state).",
    )
    p.add_argument("--save_total_limit", type=int, default=None)

    # vLLM
    p.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use TRL's built-in vLLM for online generation (recommended).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.5,
        help="vLLM GPU memory utilisation ratio.",
    )

    # Logging
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging")

    # Optional JSON config override
    p.add_argument("--config_file", type=str, default=None)

    args = p.parse_args()

    # Allow overriding from a JSON/YAML file
    if args.config_file:
        with open(args.config_file) as f:
            if args.config_file.endswith(".json"):
                override = json.load(f)
            else:
                import yaml

                override = yaml.safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)

    return args


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load Model with Unsloth ──────────────────────────────────────── #
    print(f"Loading Unsloth model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Chat template (Qwen-specific)
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # Apply LoRA via Unsloth
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

    # ── 2. Load Dataset ─────────────────────────────────────────────────── #
    print(f"Loading dataset from {args.dataset_repo} ...")
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.0)

    # ── 3. Build prompt → multiturn-data mapping ─────────────────────────── #
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    str_prompt_to_multiturn_data_map: dict[str, dict] = {}

    def process(row):
        str_prompt = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        str_prompt_to_multiturn_data_map.setdefault(
            _hash(str_prompt),
            {
                k: row[k]
                for k in [
                    "single_turn_prompt",
                    "single_turn_completion",
                    "single_turn_metadata",
                    "prompt",
                ]
            },
        )
        row["prompt"] = str_prompt
        return row

    print("Processing prompts …")
    ds["train"] = ds["train"].map(process, load_from_cache_file=False)

    # ── 4. Judge ─────────────────────────────────────────────────────────── #
    class MultiturnRewardJudge(BasePairwiseJudge):
        """Score (prompt, completion_pair) via multiturn simulation + reward."""

        def judge(self, prompts, completion_pairs, shuffle_order=False):
            rank_of_first = []
            for prompt, pair in zip(prompts, completion_pairs):
                data = str_prompt_to_multiturn_data_map.get(_hash(prompt))
                if data is None:
                    logger.warning(
                        "No multiturn data found for prompt hash; defaulting to 0"
                    )
                    rank_of_first.append(0)
                    continue

                pair_rewards = []
                for completion in pair:
                    chat_history = data["prompt"] + [
                        {"role": "assistant", "content": completion}
                    ]
                    reward_info = multiturn_aware_reward(
                        chat_history=chat_history,
                        task_desc=datasets_info[args.dataset_name]["task_desc"],
                        single_turn_prompt=data["single_turn_prompt"],
                        single_turn_completion=data["single_turn_completion"],
                        metadata=data["single_turn_metadata"],
                        metric_names=args.metric_names,
                        metric_weights=args.metric_weights,
                        user_generation_kwargs=args.user_generation_kwargs,
                        assistant_generation_kwargs=args.assistant_generation_kwargs,
                        reward_generation_kwargs=args.reward_generation_kwargs,
                        num_samples=args.num_samples,
                        max_new_turns=args.max_new_turns,
                        max_metric_workers=args.max_metric_workers,
                    )
                    pair_rewards.append(np.mean(reward_info["MR"]))

                rank_of_first.append(np.argmax(pair_rewards).item())
                logger.info(
                    f"\n[Resp 1] {pair[0][:120]}…\n"
                    f"[Resp 2] {pair[1][:120]}…\n"
                    f"Rewards: {pair_rewards}"
                )
            return torch.tensor(rank_of_first)

    judge = MultiturnRewardJudge()

    # ── 5. Training Config ───────────────────────────────────────────────── #
    report_to = "none"
    if args.wandb_project:
        report_to = "wandb"

    train_args = OnlineDPOConfig(
        beta=args.beta,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_8bit",
        report_to=report_to,
        do_eval=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=os.path.basename(args.output_dir),
        output_dir=args.output_dir,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        # vLLM
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Callbacks
    callbacks = []
    if args.use_swanlab and SWANLAB_INSTALLED:
        callbacks.append(
            SwanLabCallback(
                project=args.wandb_project or "sid-qwen3-online-dpo-unsloth",
                run_name=os.path.basename(args.output_dir),
            )
        )

    # W&B init (if applicable)
    if args.wandb_project:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=os.path.basename(args.output_dir),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )

    # ── 6. Trainer ───────────────────────────────────────────────────────── #
    trainer = OnlineDPOTrainer(
        model=model,
        judge=judge,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        args=train_args,
        callbacks=callbacks or None,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────── #
    print("Starting Online DPO training …")
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    # ── 8. Save ──────────────────────────────────────────────────────────── #
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.wandb_project:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    load_dotenv(".env")
    main()
