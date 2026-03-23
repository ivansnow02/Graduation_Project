"""
End-to-end rewrite quality test:
  1. Rewrite N DPO chosen samples via LLM API (litellm)
  2. Build dialogue files for both original & rewritten
  3. Annotate both using litellm (OpenAI-compatible, e.g. qwen-flash)
  4. Score both using objective_eval.py
  5. Print score comparison

Usage:
  uv run scripts/data_prep/rewrite_test.py -n 10 -m openai/qwen-plus --annotation-model openai/qwen-flash
"""

import argparse
import json
import sys
from pathlib import Path

import litellm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_prep.rewrite_batch import (
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_prompt,
)
from scripts.benchmark.annotation import build_prompt, extract_first_json_array
from scripts.benchmark.objective_eval import calculate_metrics


# ============================================================================
# Helpers
# ============================================================================


def build_dialogue_entry(prompt_messages: list[dict], chosen_text: str) -> dict:
    """
    Convert a DPO pair (prompt + chosen) into annotation-compatible dialogue format.
    """
    dialogue = []
    for msg in prompt_messages:
        if msg["role"] == "system":
            continue
        dialogue.append({"role": msg["role"], "content": msg["content"]})
    dialogue.append({"role": "assistant", "content": chosen_text})
    return {"dialogue": dialogue}


def annotate_pair_litellm(dialogue_pair: list[dict], model: str) -> list | None:
    """Annotate a dialogue pair using litellm instead of LM Studio."""
    prompt = build_prompt(dialogue_pair)
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5000,
        )
        text = response.choices[0].message.content.strip()
        json_str = extract_first_json_array(text)
        if not json_str:
            return None
        return json.loads(json_str)
    except Exception as e:
        print(f"    Annotation error: {e}")
        return None


def annotate_dialogue_litellm(entry: dict, model: str) -> dict:
    """Annotate all pairs in a dialogue entry using litellm."""
    turns = entry.get("dialogue", [])
    annotations = []
    for i in range(0, len(turns) - 1, 2):
        pair = turns[i : i + 2]
        if len(pair) < 2:
            break
        ann = annotate_pair_litellm(pair, model)
        if ann:
            annotations.extend(ann)
    entry["annotations"] = annotations
    return entry


# ============================================================================
# Pipeline Steps
# ============================================================================


def step1_rewrite(
    pairs: list[dict], model: str, temperature: float, n: int
) -> list[dict]:
    """Step 1: Call LLM API to rewrite chosen texts."""
    print("\n" + "=" * 70)
    print("📝 Step 1: Rewriting chosen texts via API")
    print("=" * 70)

    results = []
    for i in range(n):
        pair = pairs[i]
        prompt_messages = pair["prompt"]
        original_chosen = pair["chosen"]

        user_prompt = build_rewrite_user_prompt(
            prompt_messages=prompt_messages,
            chosen_text=original_chosen,
        )
        full_prompt = REWRITE_SYSTEM_PROMPT + "\n\n---\n\n" + user_prompt

        student_msgs = [m for m in prompt_messages if m["role"] == "user"]
        last_student = student_msgs[-1]["content"] if student_msgs else ""
        print(f"\n  [{i + 1}/{n}] 学生: {last_student[:60]}...")
        print(f"         原文: {original_chosen[:60]}...")

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"         重写: {rewritten[:60]}...")
            results.append(
                {
                    "index": i,
                    "prompt": prompt_messages,
                    "original_chosen": original_chosen,
                    "rewritten_chosen": rewritten,
                    "rejected": pair.get("rejected", ""),
                    "score_chosen": pair.get("score_chosen"),
                    "score_rejected": pair.get("score_rejected"),
                }
            )
        except Exception as e:
            print(f"         ❌ Error: {e}")
            results.append(
                {
                    "index": i,
                    "prompt": prompt_messages,
                    "original_chosen": original_chosen,
                    "rewritten_chosen": None,
                    "error": str(e),
                }
            )

    success = len([r for r in results if r.get("rewritten_chosen")])
    print(f"\n  ✅ Rewrite done: {success}/{n} succeeded")
    return results


def step2_build_dialogues(results: list[dict], output_dir: Path):
    """Step 2: Build annotation-compatible dialogue files."""
    print("\n" + "=" * 70)
    print("📂 Step 2: Building dialogue files for annotation")
    print("=" * 70)

    original_entries = []
    rewritten_entries = []

    for r in results:
        if not r.get("rewritten_chosen"):
            continue
        original_entries.append(build_dialogue_entry(r["prompt"], r["original_chosen"]))
        rewritten_entries.append(
            build_dialogue_entry(r["prompt"], r["rewritten_chosen"])
        )

    print(
        f"  Built {len(original_entries)} original + {len(rewritten_entries)} rewritten dialogues"
    )
    return original_entries, rewritten_entries


def step3_annotate(entries: list[dict], label: str, model: str) -> list[dict]:
    """Step 3: Annotate using litellm."""
    print(f"\n  Annotating {label} ({len(entries)} dialogues)...")
    annotated = []
    for i, entry in enumerate(entries):
        print(f"    [{i + 1}/{len(entries)}] ", end="", flush=True)
        result = annotate_dialogue_litellm(entry, model)
        ann_count = len(result.get("annotations", []))
        print(f"{ann_count} annotations")
        annotated.append(result)
    return annotated


def step4_evaluate(annotated: list[dict], label: str) -> list[dict]:
    """Step 4: Run objective_eval on annotated data."""
    metrics_list = []
    for entry in annotated:
        metrics = calculate_metrics(entry)
        if metrics:
            metrics_list.append(metrics)
    print(f"  {label}: {len(metrics_list)} dialogues evaluated")
    return metrics_list


def step5_compare(orig_metrics: list, rewr_metrics: list):
    """Step 5: Print score comparison."""
    print("\n" + "=" * 70)
    print("📊 Step 5: Score Comparison (Original vs Rewritten)")
    print("=" * 70)

    if not orig_metrics or not rewr_metrics:
        print("  ⚠️ No metrics to compare")
        return

    metric_keys = [
        "StrategyDensity",
        "StrategyVariety",
        "IKT",
        "BP",
        "StructureCompleteness",
        "L3GuidanceRate",
        "CognitiveCorrectionRate",
        "TotalScore",
    ]

    weights = {
        "StrategyDensity": "15%",
        "StrategyVariety": "10%",
        "IKT": "15%",
        "BP": "15%",
        "StructureCompleteness": "15%",
        "L3GuidanceRate": "10%",
        "CognitiveCorrectionRate": "20%",
        "TotalScore": "100%",
    }

    print(
        f"\n  {'Metric':<25} {'Weight':>6} {'Original':>10} {'Rewritten':>10} {'Delta':>10}"
    )
    print("  " + "-" * 65)

    for key in metric_keys:
        o_vals = [
            m["metrics_results"][key]["score"]
            for m in orig_metrics
            if key in m.get("metrics_results", {})
        ]
        r_vals = [
            m["metrics_results"][key]["score"]
            for m in rewr_metrics
            if key in m.get("metrics_results", {})
        ]
        o_avg = sum(o_vals) / len(o_vals) if o_vals else 0
        r_avg = sum(r_vals) / len(r_vals) if r_vals else 0
        delta = r_avg - o_avg

        sign = "+" if delta > 0 else ""
        emoji = "📈" if delta > 0.001 else ("📉" if delta < -0.001 else "➖")
        w = weights.get(key, "")
        fmt = ".4f" if key == "TotalScore" else ".2%"
        print(
            f"  {emoji} {key:<23} {w:>6} {o_avg:{fmt}} {r_avg:{fmt}} {sign}{delta:{fmt}}"
        )

    # Per-sample TotalScore
    print(f"\n  --- Per-Sample TotalScore ---")
    for i in range(min(len(orig_metrics), len(rewr_metrics))):
        o_ts = orig_metrics[i]["metrics_results"].get("TotalScore", {}).get("score", 0)
        r_ts = rewr_metrics[i]["metrics_results"].get("TotalScore", {}).get("score", 0)
        delta = r_ts - o_ts
        sign = "+" if delta > 0 else ""
        emoji = "✅" if delta > 0.001 else ("❌" if delta < -0.001 else "➖")
        print(f"  {emoji} Sample {i + 1}: {o_ts:.4f} → {r_ts:.4f} ({sign}{delta:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end rewrite quality test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-n", type=int, default=10, help="Number of samples")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/rewrite_batch/dpo_pairs.json",
        help="DPO pairs JSON file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai/qwen-plus",
        help="Rewrite model (litellm format)",
    )
    parser.add_argument(
        "--annotation-model",
        type=str,
        default="openai/qwen-flash",
        help="Annotation model (litellm format, default: openai/qwen-flash)",
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/rewrite_batch/test",
        help="Output directory for test results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DPO pairs
    print(f"Loading DPO pairs from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    n = min(args.n, len(pairs))
    print(
        f"Testing {n} samples | Rewrite: {args.model} | Annotation: {args.annotation_model}"
    )

    # Step 1: Rewrite
    rewrite_results = step1_rewrite(pairs, args.model, args.temperature, n)

    with open(output_dir / "rewrite_results.json", "w", encoding="utf-8") as f:
        json.dump(rewrite_results, f, ensure_ascii=False, indent=2)

    # Step 2: Build dialogues
    original_entries, rewritten_entries = step2_build_dialogues(
        rewrite_results, output_dir
    )

    # Step 3: Annotate (both original and rewritten)
    print("\n" + "=" * 70)
    print(f"🏷️  Step 3: Annotating with {args.annotation_model}")
    print("=" * 70)

    orig_annotated = step3_annotate(original_entries, "Original", args.annotation_model)
    rewr_annotated = step3_annotate(
        rewritten_entries, "Rewritten", args.annotation_model
    )

    # Save annotated data
    with open(output_dir / "original_annotated.json", "w", encoding="utf-8") as f:
        json.dump(orig_annotated, f, ensure_ascii=False, indent=2)
    with open(output_dir / "rewritten_annotated.json", "w", encoding="utf-8") as f:
        json.dump(rewr_annotated, f, ensure_ascii=False, indent=2)

    # Step 4: Evaluate
    print("\n" + "=" * 70)
    print("📏 Step 4: Running objective evaluation")
    print("=" * 70)

    orig_metrics = step4_evaluate(orig_annotated, "Original")
    rewr_metrics = step4_evaluate(rewr_annotated, "Rewritten")

    with open(output_dir / "original_metrics.json", "w", encoding="utf-8") as f:
        json.dump(orig_metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "rewritten_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rewr_metrics, f, ensure_ascii=False, indent=2)

    # Step 5: Compare
    step5_compare(orig_metrics, rewr_metrics)

    print(f"\n📁 All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
