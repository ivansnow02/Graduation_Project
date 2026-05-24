#!/usr/bin/env python3
"""生成 DPO 数据，输出为 collabllm 的嵌套格式。

- 使用来自 `SID_benchmark/output/dialog/qwen-4b` 的基线对话（chosen）
- 使用 SFT 模型或 API 生成被拒绝的对照回复（rejected）
- 输出为与 `collabllm.datasets.MultiturnDataset` 兼容的嵌套 JSON
"""

import argparse
import glob
import json
import os
from copy import deepcopy

import requests

# import torch  # 如需本地生成，可取消注释并使用本地 model+tokenizer
from tqdm import tqdm


SYSTEM_PROMPT = """你是一位跨学科的教师，始终使用苏格拉底式提问法来引导学生。你的目标不是直接给出答案，而是通过逐轮递进、循序渐进的问题，引导学生独立思考并构建跨学科理解。
你的行为规范如下：
1. 每一轮只能提出一个简洁的问题；必须体现认知推进；不能重复提问；禁止多个并列问题。
2. 你应从学生上一次回答中提炼关键点，沿着一个核心问题主线继续深入。
3. 你应始终鼓励学生跨学科思考（生物、地理、物理、历史等）。
4. 当你判断学生理解已经完成，请生成一句简洁明了的总结，明确指出学生已经完成推理，并标注[结束]。
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="outputs/sid_unsloth_sft",
        help="Path to the SFT model used to generate rejected responses (local fallback)",
    )
    parser.add_argument(
        "--input_dir",
        default="SID_benchmark/output/dialog/qwen-4b",
        help="Directory containing baseline dialogues (JSON files)",
    )
    parser.add_argument(
        "--output_file",
        default="datasets/sid_collabllm/sid_dpo_nested.json",
        help="Output path for the nested DPO dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="(可选) 限制生成的 DPO 对数。如设为 100，则生成 100 个对话轮次作为 rejected 样本后停止。用于快速测试流程而无需处理全部数据。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for rejected generation",
    )
    parser.add_argument(
        "--api_base",
        default="http://198.18.0.1:1234/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint for generating rejected",
    )
    parser.add_argument(
        "--api_model",
        default="qwen3-4b-instruct-2507-sid",
        help="Model name sent to the API",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional API key for the endpoint",
    )
    return parser.parse_args()


def load_baseline_dialogues(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            yield path, data
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")


def build_rejected_api(api_base, api_model, api_key, messages, max_new_tokens):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_new_tokens,
    }
    resp = requests.post(api_base, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response: {data}") from e


# def build_rejected_local(model, tokenizer, messages, max_new_tokens):
#     templated = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(templated, return_tensors="pt", add_special_tokens=False).to(
#         model.device
#     )
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=0.0,
#             top_p=1.0,
#             repetition_penalty=1.0,
#         )
#     gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
#     text = tokenizer.decode(gen_ids, skip_special_tokens=True)
#     return text.strip()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 选择生成后端：优先使用 API，若未提供 api_base 则回退到本地模型
    use_api = bool(args.api_base)
    model = tokenizer = None

    print(f"[INFO] API 模式: {use_api}")
    if use_api:
        print(f"[INFO] API endpoint: {args.api_base}")
        print(f"[INFO] API model: {args.api_model}")
    print(f"[INFO] 最多生成 {args.max_samples or '∞'} 个 DPO 对")
    print()

    nested_data = []
    total_pairs = 0
    total_conversations = 0

    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}

    for file_path, dialogues in load_baseline_dialogues(args.input_dir):
        print(f"[PROCESSING] {os.path.basename(file_path)} ({len(dialogues)} 个对话)")
        for idx, dialog in tqdm(
            enumerate(dialogues), total=len(dialogues), desc="Dialogues", leave=False
        ):
            dialogue = dialog.get("dialogue", [])
            if not dialogue:
                continue

            conv_id = f"{os.path.splitext(os.path.basename(file_path))[0]}__{idx}"
            single_turn_prompt = None
            turns = []
            history = []

            # metadata
            metadata = {
                "student_id": dialog.get("student_id"),
                "student_type": dialog.get("student_type"),
                "scenario": dialog.get("scenario"),
                "source": "sid_baseline_qwen4b",
            }

            for turn in dialog["dialogue"]:
                role_cn = turn.get("role")
                content = (turn.get("content") or "").strip()
                if not content:
                    continue

                if role_cn == "学生":
                    history.append({"role": "user", "content": content})
                    if single_turn_prompt is None:
                        single_turn_prompt = content
                elif role_cn == "教师":
                    # 需要至少一条用户消息，教师回复前必须有用户历史
                    if not history:
                        history.append({"role": "assistant", "content": content})
                        continue

                    prompt_msgs = deepcopy(history)

                    try:
                        if use_api:
                            rejected = build_rejected_api(
                                args.api_base,
                                args.api_model,
                                args.api_key,
                                [sys_msg] + prompt_msgs,
                                max_new_tokens=args.max_new_tokens,
                            )
                        # else:
                        #     rejected = build_rejected_local(
                        #         model,
                        #         tokenizer,
                        #         [sys_msg] + prompt_msgs,
                        #         max_new_tokens=args.max_new_tokens,
                        #     )
                    except Exception as e:
                        print(f"[WARN] 生成 rejected 回复失败 ({conv_id}): {e}")
                        continue

                    turn_entry = {
                        "prompt": prompt_msgs,
                        "responses": [
                            {"completion": content, "score": 1.0, "label": "chosen"},
                            {"completion": rejected, "score": 0.0, "label": "rejected"},
                        ],
                    }
                    turns.append(turn_entry)
                    total_pairs += 1

                    history.append({"role": "assistant", "content": content})

                    if args.max_samples and total_pairs >= args.max_samples:
                        break
                else:
                    # Unknown role; skip
                    continue

            if not turns:
                continue

            nested_data.append({
                "conv_id": conv_id,
                "single_turn_prompt": single_turn_prompt or "",
                "single_turn_completion": "",
                "single_turn_metadata": metadata,
                "turns": turns,
            })
            total_conversations += 1

            if args.max_samples and total_pairs >= args.max_samples:
                break

        if args.max_samples and total_pairs >= args.max_samples:
            break

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(nested_data, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("[DONE] DPO 数据生成完毕")
    print(f"  总 DPO 对数: {total_pairs}")
    print(f"  总对话数: {total_conversations}")
    print(f"  输出文件: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
