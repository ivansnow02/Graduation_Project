"""
运行说明：
需要：
    - 一个实现于 `examples/single_turn_ds` 下的数据集类
    - （可选）任何自定义的 metric，放在 `examples/metrics`

示例用法见脚本注释中的命令行参数示例
"""

import argparse
import hashlib
import json
import os
import os.path as osp
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv
import warnings
import random
import concurrent.futures

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.synthetic import generate_multiturn_dataset
from examples.single_turn_ds import datasets_info

# `examples.metrics` 已归档；`teaching_quality` 由 `collabllm.metric.py` 自动注册
try:
    from examples.metrics import *  # noqa: F401, F403
except ImportError:
    pass  # 旧指标已归档，主流程只使用 `teaching_quality`


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def data_engine(args):
    dataset_cls = datasets_info[args.dataset_name]["class"]
    task_desc = datasets_info[args.dataset_name]["task_desc"]
    dataset = dataset_cls().to_hf_dataset()
    # 对数据集进行打乱，以保证主题和学生类型的多样性覆盖
    shuffled_train = dataset["train"].shuffle(seed=42)

    repeat_sampling = False
    if args.train_size > 0:
        max_available = len(shuffled_train)
        if args.train_size > max_available:
            if args.allow_repeat_samples:
                warnings.warn(
                    f"Requested train_size={args.train_size} exceeds available train samples "
                    f"({max_available}). Repeating samples to reach target size.",
                    UserWarning,
                )
                rng = random.Random(42)
                indices = [
                    rng.choice(range(max_available)) for _ in range(args.train_size)
                ]
                train = shuffled_train.select(indices)
                repeat_sampling = True
            else:
                warnings.warn(
                    f"Requested train_size={args.train_size} exceeds available train samples "
                    f"({max_available}). Clipping to {max_available}.",
                    UserWarning,
                )
                train_size = min(args.train_size, max_available)
                train = shuffled_train.select(range(train_size))
        else:
            train = shuffled_train.select(range(args.train_size))
    else:
        train = shuffled_train

    # 记录分布以便验证
    student_types_count = {}
    topics_count = set()
    for item in train:
        meta = item.get("single_turn_metadata", {})
        st = meta.get("student_type", "unknown")
        topic = meta.get("topic_id", "unknown")
        student_types_count[st] = student_types_count.get(st, 0) + 1
        topics_count.add(topic)

    print(f"Selected {len(train)} samples.")
    print(f"Student Type Distribution: {student_types_count}")
    print(f"Number of Unique Topics: {len(topics_count)}")

    if args.user_prompt_file:
        with open(args.user_prompt_file, "r", encoding="utf-8") as f:
            user_prompt_template = f.read()
            args.user_generation_kwargs["prompt_template"] = user_prompt_template

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = osp.join(args.output_dir, f"{args.dataset_name}_multiturn.json")

    data_list: List[Dict[str, Any]] = []
    seen_prompt_hashes = set()

    if osp.exists(output_path):
        if args.resume:
            with open(output_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            if repeat_sampling:
                seen_prompt_hashes = {
                    compute_hash(
                        f"{ex['single_turn_prompt']}__{ex.get('sample_id', '0')}"
                    )
                    for ex in data_list
                }
            else:
                seen_prompt_hashes = {
                    compute_hash(ex["single_turn_prompt"]) for ex in data_list
                }
        else:
            warnings.warn(
                "Output file already exists. Use --resume to continue from the last saved state.",
                UserWarning,
            )
            return

    # 通过简单计数过滤（跳过已生成的样本）
    current_count = len(data_list)
    if current_count >= args.train_size:
        print(f"Already have {current_count} samples (>= {args.train_size}). Exiting.")
        return

    print(
        f"Resuming: Found {current_count} samples. Generating {args.train_size - current_count} more..."
    )

    pending_examples = []
    # 假设 'train' 可复现（已固定随机种子）。我们直接跳过前 `current_count` 个样本。
    for idx, ex in enumerate(train):
        if idx < current_count:
            continue

        if repeat_sampling:
            # 使用全局索引作为 ID，确保唯一性和可追踪性
            sample_id = f"{idx:06d}"
            pending_examples.append((sample_id, ex))
        else:
            pending_examples.append(ex)

    if not pending_examples:
        print("No new examples to generate.")
        return

    # 使用 ThreadPoolExecutor 并发生成（线程数由 --max_gen_workers 控制）
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_gen_workers
    ) as executor:
        future_to_hash = {}
        for example in pending_examples:
            if repeat_sampling:
                sample_id, example = example
                prompt_hash = compute_hash(
                    f"{example['single_turn_prompt']}__{sample_id}"
                )
            else:
                sample_id = None
                prompt_hash = compute_hash(example["single_turn_prompt"])

            # 提交任务：使用 kwargs 调用 generate_multiturn_dataset
            future = executor.submit(
                generate_multiturn_dataset,
                task_desc=task_desc,
                single_turn_prompt=example["single_turn_prompt"],
                single_turn_completion=example["single_turn_completion"],
                single_turn_metadata=example["single_turn_metadata"],
                metric_names=args.metric_names,
                user_generation_kwargs=args.user_generation_kwargs,
                assistant_generation_kwargs=args.assistant_generation_kwargs,
                reward_generation_kwargs=args.reward_generation_kwargs,
                metric_weights=args.metric_weights,
                proact_prompt_ratio=args.proact_prompt_ratio,
                num_candidate_responses=args.num_candidate_responses,
                max_total_turns=args.max_total_turns,
                max_new_turns=args.max_new_turns,
                num_samples=args.num_samples,
                max_workers=min(args.num_samples, 4),
                max_metric_workers=args.max_metric_workers,
                add_system_prompt_ratio=args.add_system_prompt_ratio,
            )

            future_to_hash[future] = prompt_hash

        # 使用 tqdm 显示并发任务完成进度
        for future in tqdm(
            concurrent.futures.as_completed(future_to_hash),
            total=len(future_to_hash),
            desc="Generating multi-turn conversations",
        ):
            prompt_hash = future_to_hash[future]
            try:
                multiturn_data = future.result()
            except Exception as e:
                print(f"Error generating for prompt {prompt_hash}: {e}")
                continue

            if multiturn_data is None:
                continue

            if repeat_sampling and sample_id is not None:
                multiturn_data["sample_id"] = sample_id

            data_list.append(multiturn_data)
            seen_prompt_hashes.add(prompt_hash)

            # 每生成一条对话后写入 JSON 文件以防止数据丢失
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)

            # 如果提供了 HF 实体（组织/用户名），则增量推送到 Hugging Face Hub
            if args.hf_entity:
                try:
                    MultiturnDataset(data_list).push_to_hub(
                        repo_id=f"{args.hf_entity}/collabllm-multiturn-{args.dataset_name}"
                    )
                except Exception as e:
                    print(f"Warning: Failed to push to Hugging Face Hub: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-turn synthetic conversations with metrics."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the single-turn dataset.",
    )
    parser.add_argument(
        "--metric_names",
        nargs="+",
        required=True,
        help="List of evaluation metric names.",
    )
    parser.add_argument(
        "--user_generation_kwargs",
        type=json.loads,
        default="{}",
        help="JSON dict of generation kwargs for user.",
    )
    parser.add_argument(
        "--user_prompt_file",
        type=str,
        default=None,
        help="Path to custom user prompt file.",
    )
    parser.add_argument(
        "--assistant_generation_kwargs",
        type=json.loads,
        default="{}",
        help="JSON dict of generation kwargs for assistant.",
    )
    parser.add_argument(
        "--reward_generation_kwargs",
        type=json.loads,
        default="{}",
        help="Optional JSON dict for reward generation.",
    )
    parser.add_argument(
        "--metric_weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights for each metric.",
    )
    parser.add_argument(
        "--proact_prompt_ratio",
        type=float,
        default=0.5,
        help="0 for none, 1 for proact, 0~1 for mixed.",
    )
    parser.add_argument(
        "--add_system_prompt_ratio",
        type=float,
        default=0,
        help="0 for none, 1 for proact, 0~1 for mixed.",
    )
    parser.add_argument(
        "--num_candidate_responses",
        type=int,
        default=2,
        help="Number of assistant candidates per turn.",
    )
    parser.add_argument(
        "--max_total_turns",
        type=int,
        default=8,
        help="Maximum number of conversation turns.",
    )
    parser.add_argument(
        "--max_new_turns",
        type=int,
        default=2,
        help="Window size for context in multi-turn generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Sample size for generating multiple conversations in one batch.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=500,
        help="Number of conversations to generate.",
    )
    parser.add_argument(
        "--allow_repeat_samples",
        action="store_true",
        help="Allow repeating training samples when train_size exceeds available data.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of parallel workers for sampling conversations.",
    )
    parser.add_argument(
        "--max_metric_workers",
        type=int,
        default=16,
        help="Maximum number of parallel workers for metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated output.",
    )
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face user or organization for dataset upload (optional).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help="Save intermediate results every N steps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last saved state if available.",
    )
    parser.add_argument(
        "--max_gen_workers",
        type=int,
        default=8,
        help="Maximum number of threads to use for generating conversations (ThreadPool size).",
    )

    load_dotenv(".env")

    args = parser.parse_args()

    print(args)
    data_engine(args)
