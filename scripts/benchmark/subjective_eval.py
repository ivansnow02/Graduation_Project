import concurrent.futures
import datetime
import glob
import json
import logging
import os
import re

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict

load_dotenv()
# requests.packages.urllib3.disable_warnings()


def resolve_path(path_value: str) -> str:
    p = Path(path_value or "")
    p = p.expanduser()
    try:
        return str(p.resolve(strict=False))
    except Exception:
        return str(p.absolute())


PROJECT_ROOT = resolve_path(str(Path(__file__).resolve().parent.parent))


def normalize_base_path(raw: str) -> str:
    if raw and raw.strip():
        return resolve_path(raw)
    return PROJECT_ROOT


def normalize_judge_api_url(raw: str) -> str:
    trimmed = (raw or "").strip()
    if not trimmed:
        return ""
    parsed = urlparse(trimmed)
    if not parsed.scheme:
        parsed = urlparse(f"http://{trimmed}")
    if not parsed.netloc:
        raise RuntimeError(f"Invalid JUDGE_API_URL: {raw}")
    return urlunparse(parsed)


BASE_PATH = normalize_base_path(os.getenv("BASE_PATH", ""))
API_URL = normalize_judge_api_url(os.getenv("JUDGE_API_URL", ""))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v3-0324").strip()
MODEL_PREFIX = (
    JUDGE_MODEL.split("/")[0] if "/" in JUDGE_MODEL else JUDGE_MODEL.split("-")[0]
)

logging.basicConfig(
    filename="evaluation_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_with_api(prompt, max_retries=3):
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    data = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一位严谨的评估专家，请严格按照用户要求输出评估结果。",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["\n\n", "###", "```", "</s>", "输出要求", "评分理由"],
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            if not API_URL:
                raise RuntimeError(
                    "Missing JUDGE_API_URL (environment variable). Please configure .env."
                )
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(data),
                verify=False,
                timeout=300,
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                if attempt < max_retries - 1:
                    continue
        except RequestException as e:
            logging.error(f"API request exception: {str(e)}")
            if attempt < max_retries - 1:
                continue
    logging.error(
        f"API call failed: Exceeded the maximum number of retries ({max_retries})"
    )
    return None


def load_dataset(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        print(f"Failed to load dataset: {str(e)}")
        return None


def build_evaluation_prompts(dataset):
    prompts = []
    metadata = []
    for item in dataset:
        dialogue_text = "\n".join([
            f"{turn['role']}: {turn['content']}" for turn in item["dialogue"]
        ])
        prompt = f"""
你是一位教育专家，正在评估以下教师与学生之间的跨学科对话质量。请根据五个关键指标进行评测：
=== 对话场景 ===
学生ID: {item["student_id"]}
学生类型: {item["student_type"]}
场景: {item["scenario"]}
学科主题: {item["topic_id"]}
=== 对话内容 ===
{dialogue_text}
=== 评测指标说明 ===
1. X-SRG（跨学科脚手架引导评分）:
   - 5: 多轮追问+逐层引导，无直接给答案
   - 4: 2轮以上引导但偶有简略
   - 3: 1轮引导未形成完整路径
   - 2: 直接陈述知识
   - 1: 教师主导无引导

2. M-RCC（多学科推理链条完整性）:
   - 5: 学科A→B→C层次清晰
   - 4: 覆盖2学科但环节跳跃
   - 3: 部分推理未闭环
   - 2: 推理链断裂
   - 1: 无推理链

3. X-MSR（跨学科错误迁移识别与修复）:
   - 5: 精准发现并使用澄清策略
   - 4: 察觉但修正不充分
   - 3: 识别但未反馈
   - 2: 忽视错误
   - 1: 未发现错误

4. CTRA（跨学科推理连接）:
   - 5: 自然迁移学科结论
   - 4: 转化生硬
   - 3: 潜在联系未显性
   - 2: 无过渡切换
   - 1: 完全割裂

5. TCF（学科过渡流畅度）:
   - 5: 学科过渡通过提问、类比、因果等手段自然发生，语言流畅
   - 4: 有过渡语言但略显模板化或节奏跳跃
   - 3: 过渡存在但略突兀，需要学生自行补逻辑
   - 2: 明显跳转，无解释、无语言承接
   - 1: 教师突然切换主题，造成学生困惑

=== 输出要求 ===
请严格按以下纯JSON格式输出结果（不要包含任何额外文本或代码块标记）：
{{
    "X-SRG": {{"score": int, "reason": "不超过50字的理由"}},
    "M-RCC": {{"score": int, "reason": "不超过50字的理由"}},
    "X-MSR": {{"score": int, "reason": "不超过50字的理由"}},
    "CTRA": {{"score": int, "reason": "不超过50字的理由"}},
    "TCF": {{"score": int, "reason": "不超过50字的理由"}}
}}

重要注意事项：
1. 输出必须是纯JSON格式，不要包含任何额外文本
2. 不要使用代码块标记(如```json)
3. 确保JSON格式完全正确（引号、括号等）
4. 评分必须是1-5的整数
5. 不要添加任何解释或说明
"""
        prompts.append(prompt)
        metadata.append({
            "student_id": item["student_id"],
            "student_type": item["student_type"],
            "scenario": item["scenario"],
            "topic_id": item["topic_id"],
            "repeat_id": item["repeat_id"],
            "dialogue": item["dialogue"],  # 保存原始对话内容
        })

    return prompts, metadata


def parse_model_response(response):
    required_keys = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    default_value = {"score": None, "reason": "解析失败"}

    if response is None:
        return {key: default_value.copy() for key in required_keys}

    cleaned = response.strip()
    result = {}
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            no_code_blocks = re.sub(
                r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE
            ).strip()
            result = json.loads(no_code_blocks)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r"\{[\s\S]*\}", cleaned)
                if json_match:
                    result = json.loads(json_match.group(0))
            except Exception:
                pass
    for key in required_keys:
        if key not in result:
            result[key] = default_value.copy()
        elif not isinstance(result[key], dict):
            result[key] = default_value.copy()
        else:
            if "score" not in result[key]:
                result[key]["score"] = None
            if "reason" not in result[key]:
                result[key]["reason"] = "理由缺失"
    return result


def evaluate_single_dialogue(prompt, meta, file_num):
    try:
        response = generate_with_api(prompt)
        eval_result = parse_model_response(response)

        return {**meta, "evaluation": eval_result, "source_file": f"topic_{file_num}"}
    except Exception as e:
        logging.error(f"Error evaluating single conversation: {str(e)}")
        return {
            **meta,
            "evaluation": {
                "X-SRG": {"score": None, "reason": "处理失败"},
                "M-RCC": {"score": None, "reason": "处理失败"},
                "X-MSR": {"score": None, "reason": "处理失败"},
                "CTRA": {"score": None, "reason": "处理失败"},
                "TCF": {"score": None, "reason": "处理失败"},
            },
            "source_file": f"topic_{file_num}",
        }


def evaluate_dialogues(dataset, file_num, max_workers=4):
    if not dataset:
        return []
    prompts, metadata = build_evaluation_prompts(dataset)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single_dialogue, prompt, meta, file_num)
            for prompt, meta in zip(prompts, metadata)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Processing {file_num}",
        ):
            results.append(future.result())
    return results


def generate_single_report(results, file_num, dataset_prefix: str = ""):
    if not results:
        return None
    metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    avg_scores = {m: 0.0 for m in metrics}
    valid_counts = {m: 0 for m in metrics}
    score_dist = {m: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for m in metrics}

    for res in results:
        for m in metrics:
            score = res["evaluation"][m]["score"]
            if isinstance(score, int) and 1 <= score <= 5:
                avg_scores[m] += score
                valid_counts[m] += 1
                score_dist[m][score] += 1
    for m in metrics:
        if valid_counts[m] > 0:
            avg_scores[m] = round(avg_scores[m] / valid_counts[m], 2)
        else:
            avg_scores[m] = 0.0
    report = {
        "file_num": file_num,
        "total_dialogues": len(results),
        "average_scores": avg_scores,
        "score_distribution": score_dist,
        "detailed_results": results,
    }
    # output path
    # If SUBJECTIVE_REPORT_DIR is provided, use it as-is.
    # Otherwise, default to: PROJECT_ROOT/SID_benchmark/output/subjective/<MODEL_PREFIX>/<dataset_prefix>
    env_report_dir = os.getenv("SUBJECTIVE_REPORT_DIR", "")
    if env_report_dir.strip():
        report_dir = resolve_path(env_report_dir)
    else:
        base_dir = os.path.join(
            PROJECT_ROOT, "SID_benchmark", "output", "subjective", MODEL_PREFIX
        )
        report_dir = (
            os.path.join(base_dir, dataset_prefix) if dataset_prefix else base_dir
        )
        report_dir = resolve_path(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    model_name = MODEL_PREFIX
    # restore json
    # Include dataset_prefix in file name to avoid collisions when multiple datasets share topic ids
    suffix = (
        f"_{dataset_prefix}" if (dataset_prefix and not env_report_dir.strip()) else ""
    )
    report_path = os.path.join(
        report_dir, f"eval_topic{file_num}_{model_name}{suffix}.json"
    )
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report file {file_num} saved to: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Failed: {str(e)}")
        return None


def generate_overall_report(all_reports, dataset_prefix: str = ""):
    if not all_reports:
        return None
    metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    overall_summary: Dict[str, Any] = {
        "total_files": len(all_reports),
        "total_dialogues": 0,
        "overall_average_scores": {m: 0.0 for m in metrics},
        "file_summaries": [],  # type: List[Dict[str, Any]]
        "overall_score_distribution": {
            m: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for m in metrics
        },
    }
    for file_num, report in all_reports.items():
        if not report:
            continue
        file_summary = {
            "file_num": file_num,
            "total_dialogues": report.get("total_dialogues", 0),
            "average_scores": report.get("average_scores", {}),
        }
        overall_summary["file_summaries"].append(file_summary)
        overall_summary["total_dialogues"] += report.get("total_dialogues", 0)

        for m in metrics:
            if m in report.get("average_scores", {}):
                overall_summary["overall_average_scores"][m] += (
                    report["average_scores"][m] * report["total_dialogues"]
                )

            if m in report.get("score_distribution", {}):
                for score, count in report["score_distribution"][m].items():
                    try:
                        score_key = int(score)
                    except (TypeError, ValueError):
                        continue
                    if score_key in overall_summary["overall_score_distribution"][m]:
                        overall_summary["overall_score_distribution"][m][score_key] += (
                            count
                        )
    for m in metrics:
        if overall_summary["total_dialogues"] > 0:
            overall_summary["overall_average_scores"][m] = round(
                overall_summary["overall_average_scores"][m]
                / overall_summary["total_dialogues"],
                2,
            )
        else:
            overall_summary["overall_average_scores"][m] = 0.0
    # output path
    env_report_dir = os.getenv("SUBJECTIVE_REPORT_DIR", "")
    if env_report_dir.strip():
        report_dir = resolve_path(env_report_dir)
    else:
        base_dir = os.path.join(
            PROJECT_ROOT, "SID_benchmark", "output", "subjective", MODEL_PREFIX
        )
        report_dir = (
            os.path.join(base_dir, dataset_prefix) if dataset_prefix else base_dir
        )
        report_dir = resolve_path(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    # overall output path (include dataset_prefix when SUBJECTIVE_REPORT_DIR not set)
    suffix = (
        f"_{dataset_prefix}" if (dataset_prefix and not env_report_dir.strip()) else ""
    )
    report_path = os.path.join(report_dir, f"overall_eval_{MODEL_PREFIX}{suffix}.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        return report_path
    except Exception as e:
        logging.error(f"Overall Failed: {str(e)}")
        return None


def main():
    # dataset path
    dataset_dir = os.getenv(
        "SUBJECTIVE_DATASET_DIR",
        os.path.join(PROJECT_ROOT, "datasets", "multi_dialogue"),
    )
    dataset_dir = resolve_path(dataset_dir)
    file_pattern = os.getenv(
        "SUBJECTIVE_FILE_PATTERN",
        os.path.join(dataset_dir, "**", "*_topic_*.json"),
    )
    file_paths = glob.glob(file_pattern, recursive=True)
    if not file_paths:
        print(f"Not Found: {file_pattern}")
        return
    print(f"Found {len(file_paths)} files to process")
    # Collect reports grouped by dataset prefix (first-level subdir under dataset_dir)
    reports_by_dataset: dict[str, dict[str, dict]] = {}
    for file_path in file_paths:
        try:
            # 尝试匹配文件名中的 topic_ID 部分
            # 针对 multi_dialogue_topic_topic_1.json 这种双重前缀的情况
            filename = os.path.basename(file_path)
            match = re.search(r"topic_(.+)\.json$", filename)
            if not match:
                print(f"Unable to extract number from file path: {file_path}")
                continue
            file_num = match.group(1)
            print(f"\n{'=' * 50}")
            print(f"Processing {file_path} (Num: {file_num})")
            dataset = load_dataset(file_path)
            if not dataset:
                print(f"{file_path} Failed, Skipped")
                continue
            # Derive dataset prefix: the first path segment under dataset_dir
            try:
                rel_parts = (
                    Path(file_path)
                    .resolve()
                    .relative_to(Path(dataset_dir).resolve())
                    .parts
                )
                dataset_prefix = rel_parts[0] if len(rel_parts) > 1 else ""
            except Exception:
                dataset_prefix = ""

            results = evaluate_dialogues(dataset, file_num)
            report_path = generate_single_report(results, file_num, dataset_prefix)
            if report_path:
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                    reports_by_dataset.setdefault(dataset_prefix, {})[file_num] = (
                        report_data
                    )
                except Exception as e:
                    logging.error(f"Loading {report_path} Failed: {str(e)}")
                    reports_by_dataset.setdefault(dataset_prefix, {})[file_num] = {
                        "file_num": file_num,
                        "total_dialogues": len(results),
                        "average_scores": {},
                        "score_distribution": {},
                        "detailed_results": results,
                    }
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            print(f"Error processing file {file_path}: {str(e)}")
    if reports_by_dataset:
        for ds_prefix, reports in reports_by_dataset.items():
            overall_report_path = generate_overall_report(reports, ds_prefix)
            if overall_report_path:
                print(
                    f"\nAll files processed for dataset '{ds_prefix or 'default'}'! Overall report saved to: {overall_report_path}"
                )
                print(f"-Count: {len(reports)}")
                print("-Avg.:")
                metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
                for metric in metrics:
                    scores = [
                        report.get("average_scores", {}).get(metric, 0)
                        for report in reports.values()
                    ]
                    if scores:
                        avg = sum(scores) / len(scores)
                        print(f" - {metric}: {avg:.2f}/5.0")
            else:
                print(
                    f"\nFailed to generate the overall report for dataset '{ds_prefix or 'default'}'"
                )
    else:
        print("\nNone")


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总处理时间: {duration:.2f} 分钟")
