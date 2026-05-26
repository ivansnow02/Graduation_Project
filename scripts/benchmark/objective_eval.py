import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

TOTAL_POSSIBLE_STRATEGIES = 8
REQUIRED_INTENTS = {"引出概念", "引导推理", "引发迁移", "总结提升"}
BLOOM_MAP = {"记忆": 1, "理解": 2, "应用": 3, "分析": 4, "评价": 5, "创造": 6}
MAX_BLOOM_PROGRESSION = len(BLOOM_MAP) - 1
NORM_CAP_JUMPS = 5.0
NORM_CAP_CORRECTIONS = 3.0


def normalize_value(value: float, cap: float) -> float:
    if cap == 0:
        return 0.0
    return min(value, cap) / cap


def calculate_metrics(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    annotations = dialogue.get("annotations", [])
    if not annotations:
        return {}
    teacher_utterances = [ann for ann in annotations if ann.get("speaker") == "教师"]
    student_utterances = [ann for ann in annotations if ann.get("speaker") == "学生"]
    teacher_utterance_count = len(teacher_utterances)
    student_utterance_count = len(student_utterances)
    strategy_utterance_count = sum(
        1 for utt in teacher_utterances if utt.get("teaching_strategy")
    )
    strategy_density = (
        strategy_utterance_count / teacher_utterance_count
        if teacher_utterance_count > 0
        else 0.0
    )
    used_strategies = {
        utt.get("teaching_strategy")
        for utt in teacher_utterances
        if utt.get("teaching_strategy")
    }
    strategy_variety = (
        len(used_strategies) / TOTAL_POSSIBLE_STRATEGIES
        if TOTAL_POSSIBLE_STRATEGIES > 0
        else 0.0
    )
    total_teacher_turns = len(teacher_utterances)
    cross_disciplinary_jumps_count = sum(
        1 for ann in annotations if ann.get("discipline_transfer") == "是"
    )
    if total_teacher_turns == 0:
        interdisciplinary_transfer_rate = 0.0
    else:
        interdisciplinary_transfer_rate = (
            cross_disciplinary_jumps_count / total_teacher_turns
        )
    final_IKT_score = interdisciplinary_transfer_rate
    student_cognition_levels = [
        BLOOM_MAP[utt.get("cognitive_level")]
        for utt in student_utterances
        if utt.get("cognitive_level") in BLOOM_MAP
    ]
    if student_cognition_levels:
        bloom_progression_raw = max(student_cognition_levels) - min(
            student_cognition_levels
        )
        norm_bloom_progression = (
            bloom_progression_raw / MAX_BLOOM_PROGRESSION
            if MAX_BLOOM_PROGRESSION > 0
            else 0.0
        )
    else:
        bloom_progression_raw = 0
        norm_bloom_progression = 0.0
    teacher_intents = {
        utt.get("teacher_intent")
        for utt in teacher_utterances
        if utt.get("teacher_intent")
    }
    covered_intents_count = len(teacher_intents.intersection(REQUIRED_INTENTS))
    structure_completeness = (
        covered_intents_count / len(REQUIRED_INTENTS) if REQUIRED_INTENTS else 0.0
    )
    l3_guidance_count = sum(
        1 for utt in teacher_utterances if utt.get("teacher_guidance_level") == "L3"
    )
    l3_guidance_rate = (
        l3_guidance_count / teacher_utterance_count
        if teacher_utterance_count > 0
        else 0.0
    )
    student_cognition_states = [
        utt.get("student_cognition_state") for utt in student_utterances
    ]
    total_error_count = student_cognition_states.count("错误回答")
    successful_correction_count = 0
    for i in range(len(student_cognition_states) - 1):
        if student_cognition_states[i] == "错误回答" and student_cognition_states[
            i + 1
        ] in ["高阶思考", "清晰理解"]:
            successful_correction_count += 1
    if total_error_count == 0:
        cognitive_correction_rate = 1.0
    else:
        cognitive_correction_rate = successful_correction_count / total_error_count
    final_3C_score = cognitive_correction_rate
    total_score = (
        0.15 * strategy_density
        + 0.10 * strategy_variety
        + 0.15 * final_IKT_score
        + 0.15 * norm_bloom_progression
        + 0.15 * structure_completeness
        + 0.10 * l3_guidance_rate
        + 0.20 * final_3C_score
    )
    return {
        "dialogue_id": dialogue.get("id"),
        "metrics_results": {
            "StrategyDensity": {
                "score": strategy_density,
                "display": f"{strategy_density:.2%}",
            },
            "StrategyVariety": {
                "score": strategy_variety,
                "display": f"{strategy_variety:.2%}",
            },
            "IKT": {"score": final_IKT_score, "display": f"{final_IKT_score:.2%}"},
            "BP": {
                "raw_value": bloom_progression_raw,
                "score": norm_bloom_progression,
                "display": f"{norm_bloom_progression:.2%}",
            },
            "StructureCompleteness": {
                "score": structure_completeness,
                "display": f"{structure_completeness:.2%}",
            },
            "L3GuidanceRate": {
                "score": l3_guidance_rate,
                "display": f"{l3_guidance_rate:.2%}",
            },
            "CognitiveCorrectionRate": {
                "score": final_3C_score,
                "display": f"{final_3C_score:.2%}",
            },
            "TotalScore": {"score": total_score, "display": f"{total_score:.4f}"},
        },
    }


def process_file(file_path: str, output_dir: str) -> List[Dict[str, Any]]:
    print(f"开始处理文件：{file_path}")
    results_for_file = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    dialogue = json.loads(line)
                    results = calculate_metrics(dialogue)
                    if results:
                        results_for_file.append(results)
                except json.JSONDecodeError:
                    print(
                        f"警告：第 {i} 行 JSON 格式无效，已跳过：{line}"
                    )
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'")
    except Exception as e:
        print(f"处理文件时发生未知错误：{e}")
    if results_for_file:
        fname = os.path.basename(file_path)
        base_name, _ = os.path.splitext(fname)
        output_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(results_for_file, out_f, indent=2, ensure_ascii=False)
        print(
            f"处理完成，单文件结果已保存到：{output_path}"
        )
    else:
        print(
            "文件中未找到有效的对话数据，因此未生成单文件结果。"
        )
    return results_for_file


def generate_summary_report(all_metrics: List[Dict], summary_path: str):
    if not all_metrics:
        print("未收集到指标数据，无法生成汇总报告。")
        return
    print("正在生成汇总报告...")
    sum_of_scores = defaultdict(float)
    sum_of_raw_values = defaultdict(float)
    total_dialogues = len(all_metrics)
    for metric_data in all_metrics:
        metrics = metric_data["metrics_results"]
        for metric_name, values in metrics.items():
            if "score" in values:
                sum_of_scores[metric_name] += values["score"]
            if "raw_value" in values:
                sum_of_raw_values[f"{metric_name}_raw_value"] += values["raw_value"]
    average_scores = {
        name: total / total_dialogues for name, total in sum_of_scores.items()
    }
    average_raw_values = {
        name: total / total_dialogues for name, total in sum_of_raw_values.items()
    }
    summary_report = {
        "total_dialogues_processed": total_dialogues,
        "average_scores": {
            k: f"{v:.4f}" if k == "TotalScore" else f"{v:.2%}"
            for k, v in average_scores.items()
        },
        "average_raw_values": {k: f"{v:.2f}" for k, v in average_raw_values.items()},
    }
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        print(f"汇总报告生成成功，已保存到：{summary_path}")
    except Exception as e:
        print(f"{e}")


def main():
    parser = argparse.ArgumentParser(
        description="计算对话评测的客观指标，输出单文件结果，并生成总体汇总报告。"
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="一个或多个输入 JSON 文件名，支持通配符。",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="_overall_summary_metrics.json",
        help="指定汇总报告的输出文件名。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="保存逐文件指标结果的目录，默认使用 `--summary-file` 所在目录。",
    )
    args = parser.parse_args()
    summary_dir = os.path.dirname(args.summary_file) or "."
    output_dir = args.output_dir if args.output_dir else summary_dir
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    all_dialogue_metrics = []
    for file_path in args.files:
        results_from_file = process_file(file_path, output_dir)
        all_dialogue_metrics.extend(results_from_file)
    generate_summary_report(all_dialogue_metrics, args.summary_file)


if __name__ == "__main__":
    main()
