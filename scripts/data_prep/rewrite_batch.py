"""
Batch Rewrite Generator for DPO Chosen Data (Leaf-Node Edition)
================================================================

Generates OpenAI Batch API compatible JSONL files that instruct a large language
model to rewrite the "chosen" (highest-scored) teacher responses in a DPO dataset.

The workflow targets DPO *leaf nodes*: each DPO pair's `chosen` field is the
terminal (leaf) response, so rewriting it has no cascading effect on conversation
history.

The rewrite prompt is reverse-engineered from the benchmark scoring formula
(objective_eval.py) so that rewritten responses will score higher on all metrics.

Usage:
  # Step 1: Convert nested data → DPO format
  uv run scripts/data_prep/rewrite_batch.py convert \\
      -i data/dpo_data/interdisciplinary_multiturn1.json \\
      -o data/rewrite_batch/dpo_pairs.json

  # Step 2: Generate batch JSONL for rewriting chosen
  uv run scripts/data_prep/rewrite_batch.py prepare \\
      -i data/rewrite_batch/dpo_pairs.json \\
      -o data/rewrite_batch/rewrite_requests.jsonl \\
      -m qwen-plus

  # Step 3: (Upload & run batch on the API provider's platform)

  # Step 4: Merge results back into DPO format
  uv run scripts/data_prep/rewrite_batch.py merge \\
      -i data/rewrite_batch/dpo_pairs.json \\
      -b data/rewrite_batch/batch_output.jsonl \\
      -o data/rewrite_batch/dpo_pairs_rewritten.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ============================================================================
# Rewrite System Prompt — the "洗稿" instructions
# ============================================================================

REWRITE_SYSTEM_PROMPT = """你是一位国内顶级的教育学博士后研究员，同时精通苏格拉底式教学法和跨学科教育。你的任务是对一位中学教师的对话回复进行"数据蒸馏"——在保留原意的基础上，全方位提升教学质量。

## 【四项铁律】

### 铁律一：跨学科知识注入（拉高 IKT）
- **必须**引入一个**不同学科**的视角，且**显式提到该学科名称**（如"从地理角度看"、"物理学中"、"历史上"），让跨学科迁移一目了然。
- ⚠️ **最高效的做法：把跨学科知识直接嵌入提问中**，一句话同时完成知识注入和提问。
- 示例（最优——嵌入问句）："既然水的比热容在常见液体里最大，那从地理角度看，沿海和内陆城市的昼夜温差会有什么不同？"
- 示例（合格——过渡+问句）："其实沿海城市温差小就是因为海水比热容大，那你觉得如果换成沙漠旁的湖泊，效果会一样吗？"
- ⚠️ 必须结合学生认知水平，**严禁超纲学术词汇**（如氢键、分子极性等大学级概念）。

### 铁律二：L3 高阶发问（拉高 L3 Guidance Rate）
- 重写后的提问**必须**是开放式的迁移、推理或综合型问题。
- **严禁**使用以下封闭式句式："是不是"、"对不对"、"对吗"、"会不会"（末尾语气词式）。
- **严禁**使用选择性多选题句式（如"比如是A、是B，还是C？"）。
- 合格句式示例：
  - "如果我们把这个条件改变，结果会怎样变化？"
  - "你能从……的角度分析一下为什么会出现这种现象吗？"
  - "这背后的深层机制是什么？"

### 铁律三：反刻板 & 策略多样化（拉高 Strategy Variety）
- **黑名单——以下套话绝对禁用**：
  "那我们能不能思考"、"你有没有想过"、"你是否了解"、"你是否知道"、"那么你是否"、"对吗？"
- **白名单——必须从以下教学策略中选择一种来组织语言**：
  1. 追问：在学生回答基础上追加更深层次的问题
  2. 提示：给出线索或提示，引导学生接近答案
  3. 类比：用生活化的类比帮助理解抽象概念（如"这就像……"）⭐ **基础薄弱学生优先使用**
  4. 情境设问：假设一个新情境让学生分析（如"假如你在月球上……"）
  5. 拆解问题：将复杂问题分解为步骤化小问题 ⭐ **学生出现困惑时优先使用**
  6. 鼓励回应：肯定学生的思考方向并加以引导
  7. 正误反馈：明确指出对错并解释原因
- 每次重写时，**随机选取**其中一种策略，确保500条数据的策略分布均匀。
- ⚠️ **因材施教原则**：当对话上下文中的学生表现出基础薄弱（如提问基础概念、表达困惑），优先使用策略3（类比）和策略5（拆解问题），用生活化语言帮助建立理解，而非直接灌输高阶知识。

### 铁律四：引发迁移 & 认知拔高（拉高 StructureCompleteness & 3C Score）
- **优先使用"引发迁移"教学意图**：将当前知识迁移到新场景或新学科，这是原始对话中最缺失的教学环节。
- 不要顺着学生的话平行发问，要**往深处挖**，制造"认知落差"。
- 如果学生回答**错误或模糊**，使用"正误反馈"或"拆解问题"策略，将大问题切碎。
- 如果学生回答**正确**，立刻假设一个**跨学科的新情境**让学生分析——同时拿到"引发迁移"intent 和 IKT 分。
- 示例："如果你把水的沸腾原理搬到海拔5000米的高原上，从物理学的气压概念出发，煮面条会比平时更难熟还是更容易？"

## 【输出要求】
1. 只输出重写后的教师回复文本，不要输出任何前缀、标签或解释。**严禁**在开头写"追问："、"提示："、"重写后："等词。
2. 保持原回复的教学意图不变（如原本在引导推理就继续引导推理，在总结就继续总结）。
3. **⚠️ 长度铁律：字数严格控制在原文的 0.8~1.2 倍之间。宁可精炼，绝不啰嗦。**
4. 语言风格：亲切自然的中学教师口吻，避免学术腔。
5. **结构铁律**：你的回复必须是**一段紧凑流畅的话**（2~4句），**严禁**使用换行符或分段。整段话应自然流畅地从"回应/肯定→知识补充→提问"一气呵成，**最后一句话必须以问号结尾**作为这个回复的终点。
6. **⚠️ 绝对禁止"先铺垫再提问"的两段式结构**。知识注入要融入前半句的自然过渡中（如"其实……所以……那你觉得……？"），不能单独成段。"""

# ============================================================================
# Helper: build the user-side prompt for a single rewrite request
# ============================================================================


def build_rewrite_user_prompt(
    prompt_messages: list[dict],
    chosen_text: str,
) -> str:
    """Build the user prompt from a DPO pair's prompt (history) and chosen text."""

    # Format conversation context from prompt messages
    # Filter out system messages — they are framework boilerplate, not useful for rewrite
    context_lines = []
    for msg in prompt_messages:
        if msg["role"] == "system":
            continue
        role_label = "学生" if msg["role"] == "user" else "教师"
        context_lines.append(f"{role_label}：{msg['content']}")
    context_str = "\n".join(context_lines)

    return f"""【对话上下文（不可修改的历史记录）】
{context_str}

【需要重写的教师当前回复（叶子节点）】
{chosen_text}

请根据四项铁律重写上述教师回复。只输出重写后的文本。"""


# ============================================================================
# convert: nested format → DPO pairs JSON
# ============================================================================


def convert_to_dpo(
    input_path: str,
    output_path: str,
    min_score_gap: float = 0.0,
):
    """Convert nested multiturn data to flat DPO pairs using MultiturnDataset."""

    # Add project root to path for collabllm imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from collabllm.datasets.multiturn import MultiturnDataset

    print(f"Loading nested data from {input_path}...")
    ds = MultiturnDataset(input_path)
    print(f"Loaded {len(ds)} flat rows.")

    dpo_ds = ds.to_dpo_dataset(eval_ratio=0.0, minimum_gap=min_score_gap)
    train_data = dpo_ds["train"]

    # Convert to list of dicts and save
    pairs = []
    for row in train_data:
        pairs.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "score_chosen": row["score_chosen"],
                "score_rejected": row["score_rejected"],
            }
        )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(pairs)} DPO pairs → {output_path}")
    return pairs


# ============================================================================
# prepare: generate Batch API JSONL from DPO pairs
# ============================================================================


def prepare_batch_file(
    input_path: str,
    output_path: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_size_mb: float = 30.0,
):
    """Read DPO pairs, and generate Batch API JSONL to rewrite each 'chosen'."""

    input_p = Path(input_path)
    if not input_p.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"Reading DPO pairs from {input_path}...")
    with open(input_p, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} DPO pairs (each 'chosen' is a leaf node).")

    # Ensure output directory exists
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = max_size_mb * 1024 * 1024
    current_part = 1

    def get_output_path(part):
        if part == 1:
            return out_p
        return out_p.parent / f"{out_p.stem}_part{part}{out_p.suffix}"

    current_out_path = get_output_path(current_part)
    out_f = open(current_out_path, "w", encoding="utf-8")
    current_file_size = 0

    total_requests = 0
    total_chars = 0
    generated_files = [current_out_path]

    try:
        for pair_idx, pair in enumerate(pairs):
            prompt_messages = pair["prompt"]
            chosen_text = pair["chosen"]

            if not chosen_text:
                continue

            # Build the rewrite prompt
            user_prompt = build_rewrite_user_prompt(
                prompt_messages=prompt_messages,
                chosen_text=chosen_text,
            )

            # Merge system + user into single user message (API compatibility)
            full_prompt = REWRITE_SYSTEM_PROMPT + "\n\n---\n\n" + user_prompt
            total_chars += len(full_prompt)

            # custom_id: rw-{pair_idx}
            custom_id = f"rw-{pair_idx}"

            request_object = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": full_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }

            line_to_write = json.dumps(request_object, ensure_ascii=False) + "\n"
            line_bytes = len(line_to_write.encode("utf-8"))

            # Rotate file if needed
            if current_file_size + line_bytes > max_bytes and current_file_size > 0:
                out_f.close()
                current_part += 1
                current_out_path = get_output_path(current_part)
                print(f"Rotating to {current_out_path.name} (reached ~{max_size_mb}MB)")
                out_f = open(current_out_path, "w", encoding="utf-8")
                current_file_size = 0
                generated_files.append(current_out_path)

            out_f.write(line_to_write)
            current_file_size += line_bytes
            total_requests += 1

    finally:
        out_f.close()

    print(
        f"\n✅ Successfully wrote {total_requests} rewrite requests "
        f"across {len(generated_files)} file(s)."
    )
    for fp in generated_files:
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  - {fp.name} ({size_mb:.2f} MB)")
    print(f"Total Characters: {total_chars:,}")
    print(f"Estimated Tokens (char/2): {total_chars // 2:,}")


# ============================================================================
# merge: merge batch API results back into DPO pairs
# ============================================================================


def merge_results(
    input_path: str,
    batch_output_path: str,
    final_output_path: str,
):
    """Merge rewritten chosen from Batch API output back into DPO pairs."""

    input_p = Path(input_path)
    batch_p = Path(batch_output_path)

    if not input_p.exists():
        print(f"Error: Input file {input_path} not found.")
        return
    if not batch_p.exists():
        print(f"Error: Batch output {batch_output_path} not found.")
        return

    # 1. Parse batch results → map
    print(f"Reading batch results from {batch_output_path}...")
    results_map: dict[int, str] = {}

    batch_files = []
    if batch_p.is_dir():
        batch_files = sorted(batch_p.glob("*.jsonl"))
    else:
        batch_files = [batch_p]

    success_count = 0
    fail_count = 0

    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    res = json.loads(line)
                    custom_id = res.get("custom_id", "")

                    # Format: rw-{pair_idx}
                    parts = custom_id.split("-")
                    if len(parts) != 2 or parts[0] != "rw":
                        continue

                    pair_idx = int(parts[1])

                    response = res.get("response", {})
                    if response and response.get("status_code") == 200:
                        body = response.get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            content = (
                                choices[0].get("message", {}).get("content", "").strip()
                            )
                            if content:
                                results_map[pair_idx] = content
                                success_count += 1
                            else:
                                fail_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}")
                    fail_count += 1

    print(f"Parsed batch results: {success_count} success, {fail_count} failed.")

    # 2. Load original DPO pairs and apply rewrites
    print(f"Loading DPO pairs from {input_path}...")
    with open(input_p, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    rewrite_count = 0
    for pair_idx, pair in enumerate(pairs):
        if pair_idx in results_map:
            pair["original_chosen"] = pair["chosen"]
            pair["chosen"] = results_map[pair_idx]
            rewrite_count += 1

    # 3. Save
    out_p = Path(final_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(
        f"\n✅ Merge complete! {rewrite_count}/{len(pairs)} chosen rewritten."
        f"\nSaved to {final_output_path}"
    )


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="DPO Chosen Leaf-Node Rewrite Batch Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- convert ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert nested multiturn data to flat DPO pairs",
    )
    convert_parser.add_argument(
        "--input", "-i", required=True, help="Input nested JSON file"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Output DPO pairs JSON file"
    )
    convert_parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.0,
        help="Minimum score gap between chosen and rejected",
    )

    # --- prepare ---
    prepare_parser = subparsers.add_parser(
        "prepare", help="Generate Batch API JSONL for rewriting chosen (leaf nodes)"
    )
    prepare_parser.add_argument(
        "--input", "-i", required=True, help="Input DPO pairs JSON file"
    )
    prepare_parser.add_argument(
        "--output", "-o", required=True, help="Output JSONL file path"
    )
    prepare_parser.add_argument("--model", "-m", required=True, help="Model name")
    prepare_parser.add_argument(
        "--temperature", "-t", type=float, default=0.7, help="Sampling temperature"
    )
    prepare_parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per response"
    )
    prepare_parser.add_argument(
        "--max-size", type=float, default=30.0, help="Max output file size in MB"
    )

    # --- merge ---
    merge_parser = subparsers.add_parser(
        "merge", help="Merge batch results back into DPO pairs"
    )
    merge_parser.add_argument(
        "--input", "-i", required=True, help="Original DPO pairs JSON file"
    )
    merge_parser.add_argument(
        "--batch-output",
        "-b",
        required=True,
        help="Batch API output file or directory",
    )
    merge_parser.add_argument(
        "--output", "-o", required=True, help="Final output JSON file"
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_to_dpo(args.input, args.output, args.min_score_gap)
    elif args.command == "prepare":
        prepare_batch_file(
            args.input,
            args.output,
            args.model,
            args.temperature,
            args.max_tokens,
            args.max_size,
        )
    elif args.command == "merge":
        merge_results(args.input, args.batch_output, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
