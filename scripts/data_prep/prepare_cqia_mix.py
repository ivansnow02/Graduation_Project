#!/usr/bin/env python3
"""
从 m-a-p/COIG-CQIA 抽取高质量百科/考试问答，
转换为 MultiturnDataset flat JSONL 格式，
与现有苏格拉底 SFT 数据合并并 Shuffle。

Usage:
    uv run scripts/data/prepare_cqia_mix.py \
        --sft_data data/converted_2500.jsonl \
        --output data/sft_mixed.jsonl \
        --cqia_ratio 0.10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from collections import Counter

from datasets import load_dataset


# COIG-CQIA 中我们感兴趣的百科/考试/知识类子集
# 参考: https://huggingface.co/datasets/m-a-p/COIG-CQIA
TARGET_TASK_TYPES = {
    "百科",
    "考试",
    "知乎",
    "wikihow_zh",
    "百科问答",
    "考试题",
    "知识问答",
}

# 宽泛关键词匹配（如果 task_type 不在上面的集合里，用关键词兜底）
TARGET_KEYWORDS = ["百科", "考试", "wiki", "知识", "问答", "exam", "knowledge"]


def is_target_task(task_type: str) -> bool:
    """判断 task_type 是否属于我们想要的类别"""
    if not task_type:
        return False
    t = task_type.strip()
    if t in TARGET_TASK_TYPES:
        return True
    t_lower = t.lower()
    return any(kw in t_lower for kw in TARGET_KEYWORDS)


def cqia_to_flat_row(item: dict, idx: int) -> dict | None:
    """将 CQIA 一条数据转换为 MultiturnDataset flat 格式"""
    instruction = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()
    output = (item.get("output") or "").strip()

    # 拼接 instruction + input 作为用户问题
    if inp:
        user_content = f"{instruction}\n{inp}"
    else:
        user_content = instruction

    # 基本质量过滤
    if not user_content or not output:
        return None
    if len(output) < 10:  # 回答太短
        return None
    if len(output) > 4000:  # 回答太长，可能超 max_seq_length
        return None

    conv_id = f"cqia_{idx:06d}_{uuid.uuid4().hex[:8]}"

    return {
        "conv_id": conv_id,
        "prompt": [{"role": "user", "content": user_content}],
        "completion": output,
        "score": 1.0,  # 人工验证数据，默认高分
        "turn_id": 1,  # 单轮对话
        "single_turn_prompt": user_content,
        "single_turn_completion": output,
        "single_turn_metadata": {
            "source": "COIG-CQIA",
            "task_type": item.get("task_type", ""),
            "domain": item.get("domain", ""),
            "human_verified": item.get("human_verified", False),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="COIG-CQIA 数据抽样 & 混合")
    parser.add_argument(
        "--sft_data",
        type=str,
        required=True,
        help="现有 SFT 数据路径 (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="合并后的输出路径 (JSONL)",
    )
    parser.add_argument(
        "--cqia_ratio",
        type=float,
        default=0.10,
        help="CQIA 数据占合并后总量的比例 (默认 0.10 = 10%%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # ── 1. 加载现有 SFT 数据 ──
    print(f"📂 加载现有 SFT 数据: {args.sft_data}")
    sft_rows = []
    with open(args.sft_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sft_rows.append(json.loads(line))
    print(f"   ✅ 已加载 {len(sft_rows)} 行")

    # 统计唯一会话数
    sft_conv_ids = set(r["conv_id"] for r in sft_rows)
    n_sft_convs = len(sft_conv_ids)
    print(f"   📊 唯一会话数: {n_sft_convs}")

    # ── 2. 计算 CQIA 抽样数量 ──
    # cqia_ratio = cqia_count / (sft_count + cqia_count)
    # → cqia_count = sft_count * ratio / (1 - ratio)
    n_cqia = math.ceil(n_sft_convs * args.cqia_ratio / (1 - args.cqia_ratio))
    print(f"   🎯 CQIA 目标抽样数: {n_cqia} 条 (占合并后 {args.cqia_ratio * 100:.0f}%)")

    # ── 3. 加载 COIG-CQIA（按子集分别加载）──
    # 可用 configs: chinese_traditional, coig_pc, exam, finance, douban,
    #   human_value, logi_qa, ruozhiba, segmentfault, wiki, wikihow, xhs, zhihu
    TARGET_CONFIGS = ["exam", "wiki", "wikihow", "zhihu"]

    print("📥 从 Hugging Face 加载 m-a-p/COIG-CQIA ...")
    candidates = []
    global_idx = 0

    for config_name in TARGET_CONFIGS:
        print(f"   📂 加载子集: {config_name} ...")
        try:
            ds = load_dataset("m-a-p/COIG-CQIA", config_name, split="train")
            print(f"      ✅ {len(ds)} 条")
        except Exception as e:
            print(f"      ❌ 加载失败: {e}")
            continue

        for item in ds:
            row = cqia_to_flat_row(dict(item), global_idx)
            if row is not None:
                # 覆盖 source 中的 task_type 为 config name
                row["single_turn_metadata"]["task_type"] = config_name
                candidates.append(row)
            global_idx += 1

    print(f"   📋 质量过滤后候选总数: {len(candidates)} 条")

    if len(candidates) < n_cqia:
        print(f"   ⚠️  候选不足 {n_cqia}，使用全部 {len(candidates)} 条")
        n_cqia = len(candidates)

    # ── 5. 随机抽样 ──
    sampled = random.sample(candidates, n_cqia)
    print(f"   🎲 抽样完成: {len(sampled)} 条")

    # ── 6. 合并 & Shuffle ──
    merged = sft_rows + sampled
    random.shuffle(merged)
    print(
        f"📦 合并后总量: {len(merged)} 行 (SFT: {len(sft_rows)}, CQIA: {len(sampled)})"
    )

    # ── 7. 写出 ──
    with open(args.output, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"💾 已保存到 {args.output}")

    # 验证统计
    merged_sources = Counter()
    for r in merged:
        src = (r.get("single_turn_metadata") or {}).get("source", "unknown")
        merged_sources[src] += 1
    print(f"📊 来源分布: {dict(merged_sources)}")


if __name__ == "__main__":
    main()
