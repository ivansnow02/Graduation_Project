import argparse
import base64
import json
import os
import sys
from pathlib import Path

# 尝试从 annotation.py 导入辅助函数
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from scripts.benchmark.annotation import build_prompt, extract_first_json_array
except ImportError:
    print(
        "警告：无法从 annotation.py 导入辅助函数，请确认你是在项目根目录下运行。"
    )
    sys.exit(1)


def encode_filename(name: str) -> str:
    """将文件名编码为适合 URL 的 base64 字符串。"""
    return base64.urlsafe_b64encode(name.encode("utf-8")).decode("utf-8")


def decode_filename(encoded: str) -> str:
    """将适合 URL 的 base64 字符串解码回文件名。"""
    try:
        return base64.urlsafe_b64decode(encoded.encode("utf-8")).decode("utf-8")
    except Exception:
        return "unknown"


def prepare_batch_file(
    input_path: str,
    output_path: str,
    model: str,
    thinking: bool = False,
    thinking_budget: int = 500,
    max_size_mb: float = 6.0,
):
    """
    读取输入 JSON/JSONL（文件或目录），生成 prompt，
    并按 `max_size_mb` 分割写出适合 Batch API 的 JSONL 文件。
    """
    input_p = Path(input_path)
    if not input_p.exists():
        print(f"Error: Input path {input_path} not found.")
        return

    files = []
    if input_p.is_dir():
        files = sorted(list(input_p.glob("*.jsonl")) + list(input_p.glob("*.json")))
    else:
        files = [input_p]

    if not files:
        print(f"No JSON/JSONL files found in {input_path}")
        return

    print(f"Processing {len(files)} files from {input_path}...")

    max_bytes = max_size_mb * 1024 * 1024
    current_part = 1

    def get_output_path(part):
        p = Path(output_path)
        if part == 1:
            return p
        return p.parent / f"{p.stem}_part{part}{p.suffix}"

    current_out_path = get_output_path(current_part)
    out_f = open(current_out_path, "w", encoding="utf-8")
    current_file_size = 0

    total_requests = 0
    total_chars = 0
    generated_files = [current_out_path]

    try:
        for file_path in files:
            file_id = encode_filename(file_path.name)

            try:
                dialogues = []
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.suffix.lower() == ".jsonl":
                        for line in f:
                            if line.strip():
                                dialogues.append(json.loads(line))
                    else:
                        dialogues = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
                continue

            for idx, entry in enumerate(dialogues, start=1):
                turns = entry.get("dialogue", [])
                for i in range(0, len(turns) - 1, 2):
                    pair = turns[i : i + 2]
                    if len(pair) < 2:
                        break

                    pair_idx = i // 2 + 1
                    prompt = build_prompt(pair)

                    total_chars += len(prompt)

                    # 自定义 ID：req-{file_id}-{d_idx}-{p_idx}
                    custom_id = f"req-{file_id}-{idx}-{pair_idx}"

                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,  # 降低温度，保证输出的稳定性和一致性
                        "max_tokens": 5000,
                    }

                    if thinking:
                        body["enable_thinking"] = True
                        body["thinking_budget"] = thinking_budget
                        body["stream"] = True

                    request_object = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }

                    line_to_write = (
                        json.dumps(request_object, ensure_ascii=False) + "\n"
                    )
                    line_bytes = len(line_to_write.encode("utf-8"))

                    # 检查是否需要切分输出文件
                    if (
                        current_file_size + line_bytes > max_bytes
                        and current_file_size > 0
                    ):
                        out_f.close()
                        current_part += 1
                        current_out_path = get_output_path(current_part)
                        print(
                            f"Rotating to {current_out_path.name} (reached ~{max_size_mb}MB)"
                        )
                        out_f = open(current_out_path, "w", encoding="utf-8")
                        current_file_size = 0
                        generated_files.append(current_out_path)

                    out_f.write(line_to_write)
                    current_file_size += line_bytes
                    total_requests += 1

    finally:
        out_f.close()

    print(
        f"Successfully wrote {total_requests} requests across {len(generated_files)} files."
    )
    for f in generated_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    print(f"Total Characters: {total_chars}")
    print(f"Estimated Tokens (char/2): {int(total_chars / 2)}")


def merge_results(input_path: str, batch_output_path: str, final_output_path: str):
    """
    将 Batch API 的结果合并回原始数据结构。
    """
    input_p = Path(input_path)
    output_p = Path(final_output_path)
    batch_file = Path(batch_output_path)

    if not input_p.exists():
        print(f"Error: Input path {input_path} not found.")
        return
    if not batch_file.exists():
        print(f"Error: Batch output file {batch_output_path} not found.")
        return

    is_dir_mode = input_p.is_dir()
    if is_dir_mode:
        if output_p.exists() and not output_p.is_dir():
            print(
                f"Error: Input is directory but output {final_output_path} is a file."
            )
            return
        output_p.mkdir(parents=True, exist_ok=True)

    # 1. 解析 Batch 结果到映射表
    print(f"Reading batch results from {batch_output_path}...")
    results_map = {}  # { filename: { (d_idx, p_idx): annotations } }

    batch_files = []
    if batch_file.is_dir():
        batch_files = sorted(list(batch_file.glob("*.jsonl")))
        print(f"Found {len(batch_files)} batch output files in directory.")
    else:
        batch_files = [batch_file]

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

                    parts = custom_id.split("-")
                    # 格式：req-{file_id}-{d_idx}-{p_idx}
                    if len(parts) < 4 or parts[0] != "req":
                        continue

                    file_name = decode_filename(parts[1])
                    d_idx = int(parts[2])
                    p_idx = int(parts[3])

                    response = res.get("response", {})
                    if response and response.get("status_code") == 200:
                        body = response.get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            json_str = extract_first_json_array(content)
                            if json_str:
                                try:
                                    ann = json.loads(json_str)
                                    if file_name not in results_map:
                                        results_map[file_name] = {}
                                    results_map[file_name][(d_idx, p_idx)] = ann
                                    success_count += 1
                                except json.JSONDecodeError:
                                    fail_count += 1
                            else:
                                fail_count += 1
                    else:
                        fail_count += 1
                except Exception:
                    pass

    print(
        f"Processed total batch results: {success_count} success, {fail_count} failed."
    )

    # 2. 遍历输入并写出结果
    files = []
    if is_dir_mode:
        files = sorted(list(input_p.glob("*.jsonl")) + list(input_p.glob("*.json")))
    else:
        files = [input_p]

    for file_path in files:
        fname = file_path.name

        # 确定输出路径
        if is_dir_mode:
            target_out = output_p / fname
        else:
            target_out = output_p

        print(f"Merging {fname} -> {target_out}...")

        file_results = results_map.get(fname, {})

        try:
            dialogues = []
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix.lower() == ".jsonl":
                    for line in f:
                        if line.strip():
                            dialogues.append(json.loads(line))
                else:
                    dialogues = json.load(f)

            with open(target_out, "w", encoding="utf-8") as out:
                for idx, entry in enumerate(dialogues, start=1):
                    turns = entry.get("dialogue", [])
                    num_pairs = len(turns) // 2
                    all_ann = []

                    for p_i in range(1, num_pairs + 1):
                        ann = file_results.get((idx, p_i))
                        if ann:
                            all_ann.extend(ann)

                    entry["annotations"] = all_ann
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("Merge complete.")


def main():
    parser = argparse.ArgumentParser(description="Batch API Annotation Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare batch input JSONL file"
    )
    prepare_parser.add_argument(
        "--input", "-i", required=True, help="Input file or directory"
    )
    prepare_parser.add_argument(
        "--output", "-o", required=True, help="Output JSONL file path"
    )
    prepare_parser.add_argument("--model", "-m", required=True, help="Model name")
    prepare_parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking mode"
    )
    prepare_parser.add_argument(
        "--thinking-budget", type=int, default=512, help="Thinking budget tokens"
    )
    prepare_parser.add_argument(
        "--max-size", type=float, default=30.0, help="Max output file size in MB"
    )

    # merge
    merge_parser = subparsers.add_parser(
        "merge", help="Merge batch results back to dataset"
    )
    merge_parser.add_argument(
        "--input", "-i", required=True, help="Original Input file or directory"
    )
    merge_parser.add_argument(
        "--batch-output", "-b", required=True, help="Batch API Output file or directory"
    )
    merge_parser.add_argument(
        "--output", "-o", required=True, help="Final Output file or directory"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_batch_file(
            args.input,
            args.output,
            args.model,
            args.thinking,
            args.thinking_budget,
            args.max_size,
        )
    elif args.command == "merge":
        merge_results(args.input, args.batch_output, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
