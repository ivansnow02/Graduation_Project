#!/usr/bin/env python3
"""
将 ../datasets/annotated 下的所有 .jsonl 文件合并并转换为 CollabLLM SFT 格式的脚本。

用法示例：
python3 scripts/convert_annotated_to_sft.py \
    --root_dir ../datasets/annotated \
    --output_path ../datasets/sid_collabllm/sid_collabllm_sft.json
"""

import argparse
import hashlib
import json
import os

try:
    import jsonlines
except Exception:
    raise SystemExit("需要安装 jsonlines：pip install jsonlines")


def get_content_hash(messages):
    """计算消息列表的 MD5 哈希指纹，截取前 8 位。"""
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]


def load_all_jsonl(root_dir):
    sid_data = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir 不存在: {root_dir}")

    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            file_path = os.path.join(subdir, fname)
            print(f"Processing: {file_path}")
            try:
                with jsonlines.open(file_path, "r") as reader:
                    for obj in reader:
                        sid_data.append(obj)
            except Exception as e:
                print(f"Warning: 读取 {file_path} 时出错: {e}")
    return sid_data


def convert_sid_to_collab(sid_data):
    collab_data = []

    for entry in sid_data:
        dialogue = entry.get("dialogue", [])
        base_id = entry.get("student_id", "unknown")

        single_turn_prompt = dialogue[0]["content"] if dialogue else ""
        metadata = {
            "source": "SID",
            "student_type": entry.get("student_type"),
            "topic": entry.get("topic_text", ""),
        }

        history = []
        seen_ids = set()

        for i, turn in enumerate(dialogue):
            role = "user" if turn.get("role") == "学生" else "assistant"
            content = turn.get("content", "")

            if role == "assistant":
                if history:
                    prompt_hash = get_content_hash(history)
                    unique_conv_id = f"{base_id}_turn_{i}_{prompt_hash}"
                    if unique_conv_id in seen_ids:
                        continue
                    seen_ids.add(unique_conv_id)

                    sample = {
                        "conv_id": unique_conv_id,
                        "single_turn_prompt": single_turn_prompt,
                        "single_turn_completion": "",
                        "single_turn_metadata": metadata,
                        "turns": [
                            {
                                "prompt": list(history),
                                "responses": [
                                    {
                                        "completion": content,
                                        "score": 1.0,
                                    }
                                ],
                            }
                        ],
                    }
                    collab_data.append(sample)

            history.append({"role": role, "content": content})

    return collab_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        default="../datasets/annotated",
        help="annotated 文件夹路径（递归查找 .jsonl）",
    )
    parser.add_argument(
        "--output_path",
        default="../datasets/sid_collabllm/sid_collabllm_sft.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    sid_data = load_all_jsonl(args.root_dir)
    print(f"读取到 SID 条目: {len(sid_data)}")

    collab_data = convert_sid_to_collab(sid_data)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(collab_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！生成 CollabLLM 训练样本数: {len(collab_data)}")
    print(f"文件已保存至: {args.output_path}")


if __name__ == "__main__":
    main()
