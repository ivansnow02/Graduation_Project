import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import requests

# 加载 `.env` 文件
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # `dotenv` 不可用时忽略


# LM Studio API 配置
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1")
# 支持多个模型，用逗号分隔，例如 `model1,model2`
ANNOTATION_MODELS = [
    m.strip() for m in os.getenv("ANNOTATION_MODELS", "").split(",") if m.strip()
]
# 兼容旧的单模型变量
if not ANNOTATION_MODELS:
    single_model = os.getenv("ANNOTATION_MODEL", "").strip()
    if single_model:
        ANNOTATION_MODELS = [single_model]

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# 采样参数
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 5000
API_TIMEOUT = 300


def call_lm_studio(prompt: str, model: str) -> str:
    """调用 LM Studio API 生成回复。"""
    try:
        # 调试输出：发送请求到模型
        response = requests.post(
            f"{LM_STUDIO_API_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
            },
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        # 调试输出：已收到模型响应
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Failed to connect to LM Studio API at {LM_STUDIO_API_URL}. "
            f"Please ensure LM Studio is running and accessible."
        )
    except Exception as e:
        raise RuntimeError(f"LM Studio API error: {e}")


def build_prompt(dialogue_pair):
    header = """
你是一位教育认知标注专家，请根据下面一段教学对话，逐轮提取以下9项教学信息，并输出为结构化 JSON 格式（列表形式）。
每一轮包含教师或学生的一个发言。请不要跳过任何一轮。
"""
    spec = """
## 【需要标注的字段】：
- speaker：发言者（"教师"或"学生"）
- utterance：原始发言文本，**不能进行任何修改**
- teacher_intent：教师发言中体现的教学目的，有以下五种："引出概念"、"检测理解"、"引导推理"、"引发迁移"、"总结提升"，**学生轮为空字符串**
- teaching_strategy：教师采用的策略，如"追问"、"提示"、"类比"、"情境设问"、"拆解问题"、"鼓励回应"、"正误反馈"等，**学生轮为空字符串**
- discipline：该轮涉及的学科，如"地理"、"生物"、"物理"、"历史"，多个学科请用逗号分隔
- discipline_transfer：若当前轮相较上轮出现新的学科，引导学科间联系，请填"是"，否则填"否"
- student_cognition_state：**仅学生轮填写**，有以下几种："清晰理解"、"模糊理解"、"表达困难"、"答非所问"、"错误回答"、"高阶思考"；**教师轮为空字符串**
- teacher_guidance_level：**仅教师轮填写**，分为3级：
  - L1：封闭性问题（是/否、定义型）
  - L2：解释/理解型问题
  - L3：迁移、推理、综合型问题
  只能标注L1，L2，L3三种
- cognitive_level：请根据 Bloom 分类，选择以下之一：
  - 记忆（Remember）
  - 理解（Understand）
  - 应用（Apply）
  - 分析（Analyze）
  - 评价（Evaluate）
  - 创造（Create）
"""
    dialogue = f"""
##【对话内容】：
{dialogue_pair[0]["role"]}：{dialogue_pair[0]["content"]}
{dialogue_pair[1]["role"]}：{dialogue_pair[1]["content"]}
"""
    output_req = """
##【请输出如下结构化标注】：（严格JSON格式，请务必使用双引号）
[
  {
    "speaker": "教师",
    "utterance": "...",
    "teacher_intent": "...",
    "teaching_strategy": "...",
    "discipline": "...",
    "discipline_transfer": "...",
    "student_cognition_state": "",
    "teacher_guidance_level": "...",
    "cognitive_level": "..."
  },
  {
    "speaker": "学生",
    "utterance": "...",
    "teacher_intent": "",
    "teaching_strategy": "",
    "discipline": "...",
    "discipline_transfer": "...",
    "student_cognition_state": "...",
    "teacher_guidance_level": "",
    "cognitive_level": "..."
  },
  ...
]
请从第一轮开始逐轮标注，直到对话结束。
"""
    return (header + spec + dialogue + output_req).strip()


def extract_first_json_array(s: str) -> str | None:
    start = s.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def annotate_pair(dialogue_pair, model: str):
    raw_prompt = build_prompt(dialogue_pair)

    try:
        text = call_lm_studio(raw_prompt, model).strip()
    except Exception as e:
        print(f"Failed to call LM Studio with model {model}: {e}")
        return None
    json_str = extract_first_json_array(text)
    if not json_str:
        print(
            f"[{model}] Failed to extract JSON array, original content: ",
            repr(text[:200]),
        )
        return None
    try:
        return json.loads(json_str)
    except Exception as e:
        # 兼容 Python 风格字面量时，回退到 `ast.literal_eval`
        try:
            import ast

            return ast.literal_eval(json_str)
        except Exception:
            pass

        print(f"[{model}] JSON Failed：", e)
        print(f"[{model}] Raw JSON content:", repr(json_str))
        return None


def load_existing_results(output_path):
    if not Path(output_path).exists():
        return 0
    count = 0
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Error reading existing result file: {e}")
        return 0
    return count


def process_entry(entry, idx, models):
    """使用多个模型轮流处理单条对话记录。"""
    model = models[(idx - 1) % len(models)]
    turns = entry.get("dialogue", [])
    num_pairs = (len(turns)) // 2
    print(
        f"\n[Conv {idx}] Starting with model: {model} ({num_pairs} pairs to annotate)..."
    )

    annotations = []
    for i in range(0, len(turns) - 1, 2):
        pair = turns[i : i + 2]
        if len(pair) < 2:
            break

        pair_idx = i // 2 + 1
        # 调试输出：标注当前对话对
        ann = annotate_pair(pair, model)
        if ann:
            annotations.extend(ann)

    entry["annotations"] = annotations
    print(
        f"--- [Conv {idx}] Completed by {model}, total {len(annotations)} annotations ---"
    )
    return entry


def annotate_dialogue_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    processed_count = load_existing_results(output_path)
    if processed_count > 0:
        print(
            f"Found {processed_count} conversations processed, continuing processing from {processed_count + 1}..."
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if processed_count > 0 else "w"

    # 准备任务
    tasks = []
    for idx, entry in enumerate(dialogues, start=1):
        if idx <= processed_count:
            continue
        tasks.append((entry, idx))

    if not tasks:
        print("All conversations processed.")
        return

    if not ANNOTATION_MODELS:
        raise RuntimeError(
            "No models specified in ANNOTATION_MODELS or ANNOTATION_MODEL environment variables."
        )

    print(
        f"Starting processing with {MAX_WORKERS} workers using models: {ANNOTATION_MODELS}"
    )
    with open(output_path, mode, encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务，并传入模型列表
            futures = [
                executor.submit(process_entry, t[0], t[1], ANNOTATION_MODELS)
                for t in tasks
            ]

            # 依次等待 future，保证输出顺序稳定
            for future in futures:
                try:
                    result_entry = future.result()
                    out_file.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    out_file.flush()
                except Exception as e:
                    print(f"Error processing entry: {e}")


def batch_process(input_dir: str, output_dir: str, input_pattern: str = "*.json"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.glob(input_pattern))
    if not files:
        print(f"No files matching `{input_pattern}` were found under {input_dir}.")
        return
    for in_file in files:
        out_file = output_path / (in_file.stem + ".jsonl")
        annotate_dialogue_file(str(in_file), str(out_file))


if __name__ == "__main__":
    # 建议通过环境变量或直接在此处填写路径。
    INPUT_DIR = os.getenv("ANNOTATION_INPUT_DIR", "").strip()
    OUTPUT_DIR = os.getenv("ANNOTATION_OUTPUT_DIR", "").strip()
    if not INPUT_DIR or not OUTPUT_DIR:
        raise RuntimeError(
            "Please set ANNOTATION_INPUT_DIR and ANNOTATION_OUTPUT_DIR (or edit annotation.py __main__)."
        )
    batch_process(INPUT_DIR, OUTPUT_DIR)
