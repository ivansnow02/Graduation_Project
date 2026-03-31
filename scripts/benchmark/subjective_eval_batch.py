"""
Batch API support for subjective evaluation of dialogues.

This script prepares dialogue data for OpenAI Batch API processing and merges
the results back into the dataset, generating evaluation reports compatible with
the main subjective_eval.py workflow.

Usage:
    # Generate batch requests
    python subjective_eval_batch.py prepare \\
        --input datasets/multi_dialogue \\
        --output batch_requests.jsonl \\
        --max-size 100

    # Merge batch results and generate reports
    python subjective_eval_batch.py merge \\
        --input datasets/multi_dialogue \\
        --batch-output batch_output.jsonl \\
        --output datasets/multi_dialogue_evaluated
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

# Try to import helper functions from subjective_eval
try:
    from scripts.benchmark.subjective_eval import (
        build_evaluation_prompts,
        parse_model_response,
        generate_single_report,
        generate_overall_report,
        resolve_path,
        normalize_base_path,
        JUDGE_MODEL,
        API_URL,
        PROJECT_ROOT,
    )
except ImportError as e:
    print(f"Warning: Could not import helper functions from subjective_eval.py: {e}")
    print("Please ensure you are running from project root.")
    sys.exit(1)

load_dotenv()

logging.basicConfig(
    filename="evaluation_batch_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def encode_filename(name: str) -> str:
    """Encodes a filename to a URL-safe base64 string."""
    return base64.urlsafe_b64encode(name.encode("utf-8")).decode("utf-8")


def decode_filename(encoded: str) -> str:
    """Decodes a URL-safe base64 string back to a filename."""
    try:
        return base64.urlsafe_b64decode(encoded.encode("utf-8")).decode("utf-8")
    except Exception:
        return "unknown"


def prepare_batch_file(
    input_path: str,
    output_path: str,
    max_size_mb: float = 100.0,
) -> None:
    """
    Reads input JSON/JSONL files, generates evaluation prompts,
    and writes JSONL files suitable for Batch API, splitting at max_size_mb.

    Args:
        input_path: Directory or file path containing dialogue data
        output_path: Output JSONL file path (will be split into parts if needed)
        max_size_mb: Maximum size per output file in MB
    """
    input_p = Path(input_path)
    if not input_p.exists():
        print(f"Error: Input path {input_path} not found.")
        return

    # Collect files to process
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
            print(f"Processing {file_path.name}...")

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
                logging.error(f"Error reading {file_path.name}: {e}")
                continue

            # Build evaluation prompts for all dialogues
            try:
                prompts, metadata = build_evaluation_prompts(dialogues)
            except Exception as e:
                print(f"Error building prompts for {file_path.name}: {e}")
                logging.error(f"Error building prompts for {file_path.name}: {e}")
                continue

            # Process each dialogue
            for idx, (prompt, meta) in enumerate(zip(prompts, metadata), start=1):
                if prompt is None:
                    continue

                total_chars += len(prompt)

                # Custom ID: req-{file_id}-{dialogue_idx}
                custom_id = f"req-{file_id}-{idx}"

                body = {
                    "model": JUDGE_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位严谨的评估专家，请严格按照用户要求输出评估结果。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "stop": [
                        "\n\n",
                        "###",
                        "```",
                        "</s>",
                        "输出要求",
                        "评分理由",
                    ],
                    "stream": False,
                }

                request_object = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }

                line_to_write = json.dumps(request_object, ensure_ascii=False) + "\n"
                line_bytes = len(line_to_write.encode("utf-8"))

                # Check if we need to rotate to next file
                if current_file_size + line_bytes > max_bytes and current_file_size > 0:
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


def merge_results(
    input_path: str, batch_output_path: str, final_output_path: str
) -> None:
    """
    Merges batch API results back into the original dataset structure and
    generates evaluation reports.

    Args:
        input_path: Original input file or directory path
        batch_output_path: Batch API output file or directory path
        final_output_path: Final output file or directory path
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

    # 1. Parse Batch Results -> Map
    print(f"Reading batch results from {batch_output_path}...")
    results_map = {}  # { filename: { dialogue_idx: annotations } }

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
                    # Format: req-{file_id}-{dialogue_idx}
                    if len(parts) < 3 or parts[0] != "req":
                        continue

                    file_name = decode_filename(parts[1])
                    dialogue_idx = int(parts[2])

                    response = res.get("response", {})
                    if response and response.get("status_code") == 200:
                        body = response.get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            eval_result = parse_model_response(content)
                            if eval_result:
                                if file_name not in results_map:
                                    results_map[file_name] = {}
                                results_map[file_name][dialogue_idx] = eval_result
                                success_count += 1
                            else:
                                fail_count += 1
                        else:
                            fail_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logging.error(f"Error parsing batch result: {e}")
                    fail_count += 1

    print(f"Processed {success_count} successful results, {fail_count} failed results.")

    # 2. Iterate inputs and write outputs with evaluations
    files = []
    if is_dir_mode:
        files = sorted(list(input_p.glob("*.jsonl")) + list(input_p.glob("*.json")))
    else:
        files = [input_p]

    # Collect all results for report generation
    reports_by_dataset: Dict[str, Dict[str, Any]] = {}

    for file_path in files:
        fname = file_path.name

        # Determine output path
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

            # Process each dialogue and add evaluation
            output_data = []
            results_for_report = []

            for idx, entry in enumerate(dialogues, start=1):
                eval_result = file_results.get(idx)
                if eval_result:
                    entry["evaluation"] = eval_result
                    results_for_report.append({
                        **entry,
                        "evaluation": eval_result,
                        "source_file": f"topic_{idx}",
                    })
                else:
                    # Add empty evaluation if not found
                    entry["evaluation"] = {
                        "X-SRG": {"score": None, "reason": "未获取评估"},
                        "M-RCC": {"score": None, "reason": "未获取评估"},
                        "X-MSR": {"score": None, "reason": "未获取评估"},
                        "CTRA": {"score": None, "reason": "未获取评估"},
                        "TCF": {"score": None, "reason": "未获取评估"},
                    }

                output_data.append(entry)

            # Write merged output
            with open(target_out, "w", encoding="utf-8") as out:
                if target_out.suffix.lower() == ".jsonl":
                    for item in output_data:
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    json.dump(output_data, out, ensure_ascii=False, indent=2)

            # Generate report for this file
            if results_for_report:
                # Derive dataset prefix: first path segment under input base
                try:
                    rel_parts = (
                        Path(file_path)
                        .resolve()
                        .relative_to(Path(input_path).resolve())
                        .parts
                    )
                    dataset_prefix = rel_parts[0] if len(rel_parts) > 1 else ""
                except Exception:
                    dataset_prefix = ""

                reports_by_dataset.setdefault(dataset_prefix, {})[idx] = {
                    "file_num": idx,
                    "total_dialogues": len(results_for_report),
                    "average_scores": {},
                    "score_distribution": {},
                    "detailed_results": results_for_report,
                }

            print(f"Merged {len(output_data)} dialogues to {target_out}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            logging.error(f"Error processing {fname}: {e}")

    # 3. Generate reports
    if reports_by_dataset:
        for ds_prefix, results_dict in reports_by_dataset.items():
            try:
                # Generate single file reports and collect them
                generated_reports = {}
                for file_num, result_data in results_dict.items():
                    results = result_data.get("detailed_results", [])
                    report_path = generate_single_report(results, file_num, ds_prefix)
                    if report_path:
                        print(f"Generated report: {report_path}")
                        # Read the generated report back
                        try:
                            with open(report_path, "r", encoding="utf-8") as f:
                                report_content = json.load(f)
                                generated_reports[file_num] = report_content
                        except Exception as e:
                            logging.error(
                                f"Failed to read generated report {report_path}: {e}"
                            )

                # Generate overall report from collected reports
                if generated_reports:
                    overall_report_path = generate_overall_report(
                        generated_reports, ds_prefix
                    )
                    if overall_report_path:
                        print(
                            f"\nAll files processed for dataset '{ds_prefix or 'default'}'!"
                        )
                        print(f"Overall report saved to: {overall_report_path}")
                        print(f"Total files: {len(generated_reports)}")
                        print("-Average scores:")
                        metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]

                        # Extract and display average scores from generated reports
                        for metric in metrics:
                            scores = []
                            for report in generated_reports.values():
                                avg_score = report.get("average_scores", {}).get(metric)
                                if (
                                    isinstance(avg_score, (int, float))
                                    and 1 <= avg_score <= 5
                                ):
                                    scores.append(avg_score)
                            if scores:
                                overall_avg = sum(scores) / len(scores)
                                print(f" - {metric}: {overall_avg:.2f}/5.0")

            except Exception as e:
                print(f"Error generating reports for dataset '{ds_prefix}': {e}")
                logging.error(
                    f"Error generating reports for dataset '{ds_prefix}': {e}"
                )

    print("\nMerge and report generation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch API support for subjective evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare batch input JSONL file"
    )
    prepare_parser.add_argument(
        "--input", "-i", required=True, help="Input file or directory"
    )
    prepare_parser.add_argument(
        "--output", "-o", required=True, help="Output JSONL file path"
    )
    prepare_parser.add_argument(
        "--max-size", type=float, default=100.0, help="Max output file size in MB"
    )

    # merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge batch results back to dataset and generate reports"
    )
    merge_parser.add_argument(
        "--input", "-i", required=True, help="Original input file or directory"
    )
    merge_parser.add_argument(
        "--batch-output",
        "-b",
        required=True,
        help="Batch API output file or directory",
    )
    merge_parser.add_argument(
        "--output", "-o", required=True, help="Final output file or directory"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_batch_file(args.input, args.output, args.max_size)
    elif args.command == "merge":
        merge_results(args.input, args.batch_output, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
