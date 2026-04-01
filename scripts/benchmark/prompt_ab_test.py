"""
prompt_ab_test.py
~~~~~~~~~~~~~~~~~
用 QwenFlash 对不同 System Prompt 进行端到端 A/B 测试。

流程：生成对话 → 标注 → 打分 → 对比
用法：
    cd /Volumes/Backup/dev/Graduation_Project
    uv run scripts/benchmark/prompt_ab_test.py --topic-idx 1 --repeats 3
"""

import argparse
import json
import os
import sys
import time
import random
import logging
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import dotenv

dotenv.load_dotenv()

# litellm
import litellm
from litellm import completion

litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []

import warnings

warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("prompt_ab_test.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

# ── API 配置 ──
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
API_BASE = os.getenv("OPENAI_API_BASE", "").strip()
# 使用 qwen-flash 做教师/学生，qwen-flash 做标注
TEACHER_MODEL = "qwen-flash"
ANNOTATOR_MODEL = "qwen-flash"

# ── 截断配置（防止上下文过长导致调用或标注失败） ──
GEN_CONTEXT_MAX_TURNS = 14  # 生成时最多携带最近多少条历史消息（允许更长对话以展现SV）
ANNOTATION_MAX_TURNS = 16  # 标注时最多保留最近多少条对话（允许更长对话以展现SV）

# ── System Prompts to Test ──
# 基于 objective_eval.py 公式: Total = 0.15*SD + 0.10*SV + 0.15*IKT + 0.15*BP + 0.15*SC + 0.10*L3 + 0.20*CCR
# Baseline (原始 system_prompt.txt, 480 dialogues): SD=100%, SV=34%, IKT=52%, BP=65%, SC=57%, L3=93%, CCR=100%, Total=0.738
PROMPTS = {
    # === Chosen: 真实苏格拉底跨学科导师（自然对话流，保留高分核心） ===
    "socratic_chosen": """你是一位苏格拉底式跨学科导师，善于通过多轮对话引导学生从表面理解逐步深化到系统性思维。

教学原则：
- 从学生回答自然递进。用"你提到的..."或"既然你认识到..."这样的格式开局，而不是生硬切题。这样学生感到被倾听，对话自然流畅。
- 逐步建立机制理解。前期围绕现象提问，逐步引导思考"为什么"与"如果-那么"的因果链。每轮关键点用因果词（因为、所以、如果）明确表述关键机制。
- 自然融入跨学科视角。当学生的理解固化时，用类比或具体情境引入其他学科。从地理现象迁移到物理原理、从个案拓展到生态系统，保持对话流畅。
- 混用思维工具。追问、类比、情境设问、正误反馈等作为对话的有机部分，根据学生理解状态灵活调整，避免生硬标签化。
- 渐进式纠正。学生答错或模糊时，下一轮针对性澄清与引导，逐步推向更清晰、更高阶的表达。
- 认知递进。从"理解/应用"→"分析"→"评价/创造"，避免在同一层级反复停滞。

执行标准：
- 每轮45~110字，自然教师语言（减少"很好""不过"等填充词），每轮以开放性问题结尾。
- 至少进行5~6轮交互，在合适时机进行"总结提升"式回应，回顾跨学科迁移链条并给予展望。""",
    # === Rejected_1: 单学科、灌输式、低互动（通顺但低质） ===
    "single_discipline_rejected": """你是一位表达流畅但低互动的单学科讲授老师。
负面规范（必须严格遵守，保持语句通顺）：
1. **禁止提问**：全程不用“？”；不追问、不检测理解。
2. **单学科封闭**：只允许使用一个学科视角，禁止跨学科类比与迁移。
3. **直接灌输**：每轮直接给出答案和定义，不让学生参与推理。
4. **固定模板**：每轮严格输出三句：
   第1句“核心结论：...”；第2句“标准解释：...”；第3句“记住这个定义即可。”
5. **低阶表达**：避免因果链、比较、反例、迁移等高阶表达。
6. **无纠错支架**：学生答错时，只重复标准解释，不做针对性澄清。
7. **终止规则**：至少进行 4 轮师生交互后再结束。""",
    # === Rejected_2: 表面互动但浅层循环（通顺但低质） ===
    "shallow_loop_rejected": """你是一位表达自然但几乎无教学推进的安抚型老师。
负面规范（必须严格遵守）：
1. **禁止提问**：全程不用“？”；不得出现追问、检测理解或引导推理。
2. **禁迁移与禁跨学科词**：不出现物理/化学/生物/工程等学科词，也不做跨学科连接。
3. **浅层安抚**：每轮先安抚情绪，再给笼统建议；语句可变化，但不得引入新概念。
4. **低阶停滞**：每轮仅做“安抚+复述”，不出现因果、比较、反例等高阶表达。
5. **禁止策略变化**：不要使用类比、拆解、正误反馈、追问等教学策略。
6. **弱纠错**：学生答错时仅做泛化鼓励，不解释错误原因，也不要求其修正。
7. **终止规则**：至少进行 4 轮师生交互后再结束。""",
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_dialogue_surface(dialogue: List[Dict]) -> Dict[str, float]:
    """基于原始对话的表面质量分，减轻标注噪声影响。"""
    teacher_turns = [t for t in dialogue if t.get("role") == "教师"]
    t_count = len(teacher_turns)
    if t_count == 0:
        return {"QuestionRate": 0.0, "FillerPenalty": 0.0}

    question_count = sum(
        1
        for t in teacher_turns
        if str(t.get("content", "")).strip().endswith(("?", "？"))
    )
    question_rate = question_count / t_count

    filler_tokens = ["啊", "哦", "嗯", "哈哈"]
    filler_hits = 0
    total_chars = 0
    for t in teacher_turns:
        text = str(t.get("content", ""))
        total_chars += len(text)
        filler_hits += sum(text.count(tok) for tok in filler_tokens)

    filler_density = (filler_hits / max(total_chars, 1)) * 100.0
    filler_penalty = min(1.0, filler_density / 3.0)

    return {
        "QuestionRate": round(clamp01(question_rate), 4),
        "FillerPenalty": round(clamp01(filler_penalty), 4),
    }


def call_llm(messages, model=TEACHER_MODEL, temperature=0.7, max_tokens=2048):
    """统一 LLM 调用"""
    api_base = API_BASE
    if api_base and not api_base.endswith("/v1"):
        if "/v1" not in api_base:
            api_base = f"{api_base.rstrip('/')}/v1"

    for attempt in range(3):
        try:
            response = completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=API_KEY,
                custom_llm_provider="openai",
                temperature=temperature,
                max_tokens=max_tokens,
                drop_params=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None


def count_approx_tokens(text: str) -> int:
    """粗略估算 Token 数量 (Chinese-English mix)"""
    # 估算规则: 1 个汉字 = 1.5 tokens, 1 个英文单词/标点 = 1 token
    # 简单计算: 字符数 * 1.2
    return int(len(text) * 1.2)


def clean_llm_json(json_str: str) -> str:
    """清洗 LLM 输出的 JSON，处理未转义的换行符和非法控制字符"""
    if not json_str:
        return ""
    # 1. 替换字符串内的原始换行符为 \n
    # 这个正则寻找 "key": "value" 结构中 value 内容中包含的回车
    # 稍微暴力一点的方案：先转义，再还原合法的 JSON 结构
    # 但由于 JSON 本身就有双引号，我们可以尝试检测双引号内的原始换行
    cleaned = []
    in_string = False
    escape = False
    for char in json_str:
        if char == '"' and not escape:
            in_string = not in_string

        if in_string:
            if char == "\n":
                cleaned.append("\\n")
            elif char == "\r":
                cleaned.append("\\r")
            elif char == "\t":
                cleaned.append("\\t")
            else:
                cleaned.append(char)
        else:
            cleaned.append(char)

        if char == "\\" and not escape:
            escape = True
        else:
            escape = False

    return "".join(cleaned)


def truncate_recent_history(history: List[Dict], max_turns: int) -> List[Dict]:
    """仅保留最近 max_turns 条历史，避免上下文爆长。"""
    if max_turns <= 0:
        return history
    if len(history) <= max_turns:
        return history
    return history[-max_turns:]


def truncate_dialogue_for_annotation(
    dialogue: List[Dict], max_turns: int
) -> List[Dict]:
    """标注前对超长对话做硬截断，优先保留最近轮次。"""
    if max_turns <= 0:
        return dialogue
    if len(dialogue) <= max_turns:
        return dialogue
    return dialogue[-max_turns:]


# ── 对话生成 ──
def generate_dialogue(
    topic_text: str,
    system_prompt: str,
    max_turns: int = 7,
    prompt_name: Optional[str] = None,
    initial_question: Optional[str] = None,
    student_plan: Optional[List[Dict[str, str]]] = None,
) -> List[Dict]:
    """用指定 system_prompt 生成一段完整的多轮教学对话"""
    history = []

    # 1. 学生初始提问
    init_prompt = f"""你是一名中学生，正在学习一节跨学科课程。请根据下面的课程内容，提出一个你真正感到困惑或好奇的问题。
要求：回答要口语化，符合中学生身份。只输出问题内容。

课程内容如下：
{topic_text}"""

    question = initial_question
    if not question:
        question = call_llm([{"role": "user", "content": init_prompt}], temperature=0.9)
    if not question:
        return []
    history.append({"role": "学生", "content": question.strip()})

    student_types = [
        "你是一位基础薄弱的学生，对概念掌握不牢",
        "你是一位有一定基础的学生，善于类比推理",
        "你是一位全优型学生，逻辑清晰",
    ]
    scenarios = [
        "学生尝试回答但有部分错误",
        "学生进行了一个类比猜测",
        "学生回答正确并尝试延伸",
        "学生不太理解，请求老师解释",
    ]

    for turn in range(max_turns):
        # 1-1. 长度与 Token 预警
        current_text = str(system_prompt) + "\n".join([
            f"{h['role']}: {h['content']}" for h in history
        ])
        total_tokens = count_approx_tokens(current_text)

        # 教师回复：严格使用静态 system prompt（与实际大规模生成一致）
        teacher_msgs = [{"role": "system", "content": system_prompt}]

        # 强制熔断补丁: 如果接近 2048 限制 (1800 开始预防)
        if total_tokens > 1800:
            teacher_msgs[0]["content"] += (
                "\n**强制指令：当前对话已过长，你必须立刻总结当前知识点，并在总结最后加上 [结束] 字样。**"
            )

        recent_history = truncate_recent_history(history, GEN_CONTEXT_MAX_TURNS)
        for h in recent_history:
            role = "user" if h["role"] == "学生" else "assistant"
            teacher_msgs.append({"role": role, "content": h["content"]})

        teacher_reply = call_llm(teacher_msgs, temperature=0.9, max_tokens=512)
        if not teacher_reply:
            break

        # 强制前 3 轮不准结束 (turn 0=学生, turn 1=教师, turn 2=学生)
        # 这里 turn 从 0 开始，turn=0 是第一轮对话循环
        if "[结束]" in teacher_reply:
            if turn < 3:
                # 强行驱散结束标记，并诱导模型追问
                teacher_reply = teacher_reply.replace("[结束]", "").strip()
                if not teacher_reply.endswith("？"):
                    teacher_reply += " 对于这个跨学科的切入点，你还有什么想问的吗？"
                history.append({"role": "教师", "content": teacher_reply})
            else:
                history.append({
                    "role": "教师",
                    "content": teacher_reply.replace("[结束]", "").strip(),
                })
                break
        else:
            history.append({"role": "教师", "content": teacher_reply.strip()})

        if turn >= max_turns - 1:
            break

        # 学生回复
        if student_plan and turn < len(student_plan):
            identity = student_plan[turn].get("identity", random.choice(student_types))
            scenario = student_plan[turn].get("scenario", random.choice(scenarios))
        else:
            identity = random.choice(student_types)
            scenario = random.choice(scenarios)

        student_sys = f"""你是一名中学生。请根据教师的问题给出回答。
    你的设定：{identity}
    当前状态：{scenario}
    要求：回答要口语化，符合中学生身份。只输出回答内容。"""

        student_msgs = [{"role": "system", "content": student_sys}]
        recent_history_for_student = truncate_recent_history(
            history, GEN_CONTEXT_MAX_TURNS
        )
        for h in recent_history_for_student:
            role = "user" if h["role"] == "教师" else "assistant"
            student_msgs.append({"role": role, "content": h["content"]})

        student_reply = call_llm(student_msgs, temperature=1.0)
        if not student_reply:
            break
        history.append({"role": "学生", "content": student_reply.strip()})

    return history


# ── 标注 ──
def build_annotation_prompt(dialogue: List[Dict]) -> str:
    """构建标注 prompt（复用 annotation.py 的逻辑）"""
    header = """你是一位教育认知标注专家，请根据下面一段教学对话，逐轮提取以下9项教学信息，并输出为结构化 JSON 格式（列表形式）。
每一轮包含教师或学生的一个发言。请不要跳过任何一轮。"""

    spec = """
## 【需要标注的字段】：
- speaker：发言者（"教师"或"学生"）
- utterance：原始发言文本，**不能进行任何修改**
- teacher_intent：教师发言中体现的教学目的，有以下五种："引出概念"、"检测理解"、"引导推理"、"引发迁移"、"总结提升"，**学生轮为空字符串**
- teaching_strategy：教师采用的策略，如"追问"、"提示"、"类比"、"情境设问"、"拆解问题"、"鼓励回应"、"正误反馈"等，**学生轮为空字符串**
- discipline：该轮涉及的学科，如"地理"、"生物"、"物理"、"历史"，多个学科请用逗号分隔
- discipline_transfer：若当前轮相较上轮出现新的学科，引发了学科间知识迁移，标注为"是"；否则标注为"否"
- teacher_guidance_level：教师引导的程度，分三级 "L1"表示直接告知答案，"L2"表示给出部分提示，"L3"表示仅通过提问引导学生自主探索，**学生轮为空字符串**
- cognitive_level：学生展示的认知水平，如"记忆"、"理解"、"应用"、"分析"、"评价"、"创造"，**教师轮为空字符串**
- student_cognition_state：学生认知状态，如"模糊回答"、"清晰理解"、"高阶思考"、"错误回答"，**教师轮为空字符串**

请严格按上述字段输出 JSON 列表。"""

    dialogue_text = "\n".join([f"{t['role']}：{t['content']}" for t in dialogue])
    return f"{header}\n{spec}\n\n## 【对话内容】：\n{dialogue_text}\n\n请输出标注结果（JSON列表）："


def annotate_dialogue(dialogue: List[Dict]) -> Optional[List[Dict]]:
    """用 LLM 标注一段对话"""
    truncated_dialogue = truncate_dialogue_for_annotation(
        dialogue, ANNOTATION_MAX_TURNS
    )
    if len(truncated_dialogue) < len(dialogue):
        logger.info(
            "Annotation dialogue truncated: %s -> %s turns",
            len(dialogue),
            len(truncated_dialogue),
        )
    prompt = build_annotation_prompt(truncated_dialogue)
    result = call_llm(
        [{"role": "user", "content": prompt}],
        model=ANNOTATOR_MODEL,
        temperature=0.0,
        max_tokens=8000,
    )
    if not result:
        return None

    # 提取 JSON
    import re

    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", result, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 尝试直接找 JSON 数组
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            logger.error(f"No JSON found in annotation result: {result[:200]}")
            return None

    try:
        # 清洗 JSON 字符串
        cleaned_json = clean_llm_json(json_str)
        annotations = json.loads(cleaned_json)
        return annotations
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error (attempt 1): {e}")
        # 灾难级兜底：尝试使用 ast 处理（处理单引号或更松散的格式）
        try:
            import ast

            # 清洗后的字符串如果还是报错，尝试直接用 ast
            annotations = ast.literal_eval(cleaned_json)
            if isinstance(annotations, list):
                return annotations
        except Exception as e2:
            logger.error(f"Final failed to parse annotation: {e2}")
            return None


# ── 打分（复用 objective_eval.py 逻辑）──
TOTAL_POSSIBLE_STRATEGIES = 8
REQUIRED_INTENTS = {"引出概念", "引导推理", "引发迁移", "总结提升"}
BLOOM_MAP = {"记忆": 1, "理解": 2, "应用": 3, "分析": 4, "评价": 5, "创造": 6}
MAX_BLOOM_PROGRESSION = len(BLOOM_MAP) - 1


def score_annotations(annotations: List[Dict]) -> Dict[str, float]:
    """对标注结果打分，返回各项指标"""
    if not annotations:
        return {}

    teacher_anns = [a for a in annotations if a.get("speaker") == "教师"]
    student_anns = [a for a in annotations if a.get("speaker") == "学生"]
    t_count = len(teacher_anns)

    # 1. Strategy Density
    strategy_count = sum(1 for a in teacher_anns if a.get("teaching_strategy"))
    sd = strategy_count / t_count if t_count > 0 else 0.0

    # 2. Strategy Variety
    strategies = {
        a.get("teaching_strategy") for a in teacher_anns if a.get("teaching_strategy")
    }
    sv = (
        len(strategies) / TOTAL_POSSIBLE_STRATEGIES
        if TOTAL_POSSIBLE_STRATEGIES > 0
        else 0.0
    )

    # 3. IKT - 与 objective_eval.py 一致: 统计所有轮次 discipline_transfer / 教师轮数
    ikt_count = sum(1 for a in annotations if a.get("discipline_transfer") == "是")
    ikt = ikt_count / t_count if t_count > 0 else 0.0

    # 4. Structure Completeness
    intents = {a.get("teacher_intent") for a in teacher_anns if a.get("teacher_intent")}
    sc = (
        len(intents.intersection(REQUIRED_INTENTS)) / len(REQUIRED_INTENTS)
        if REQUIRED_INTENTS
        else 0.0
    )

    # 5. L3 Guidance Rate
    l3_count = sum(1 for a in teacher_anns if a.get("teacher_guidance_level") == "L3")
    l3 = l3_count / t_count if t_count > 0 else 0.0

    # 6. Bloom Progression
    bloom_levels = [
        BLOOM_MAP[a.get("cognitive_level")]
        for a in student_anns
        if a.get("cognitive_level") in BLOOM_MAP
    ]
    if bloom_levels:
        bp = (max(bloom_levels) - min(bloom_levels)) / MAX_BLOOM_PROGRESSION
    else:
        bp = 0.0

    # 7. CCR
    states = [a.get("student_cognition_state") for a in student_anns]
    error_count = states.count("错误回答")
    correction_count = 0
    for i in range(len(states) - 1):
        if states[i] == "错误回答":
            if states[i + 1] in ["高阶思考", "清晰理解"]:
                correction_count += 1
    ccr = 1.0 if error_count == 0 else (correction_count / error_count)

    # Total - 与 objective_eval.py 的权重完全一致
    total = (
        0.15 * sd
        + 0.10 * sv
        + 0.15 * ikt
        + 0.15 * bp
        + 0.15 * sc
        + 0.10 * l3
        + 0.20 * ccr
    )

    return {
        "StrategyDensity": round(sd, 4),
        "StrategyVariety": round(sv, 4),
        "IKT": round(ikt, 4),
        "BloomProgression": round(bp, 4),
        "StructureCompleteness": round(sc, 4),
        "L3GuidanceRate": round(l3, 4),
        "CognitiveCorrectionRate": round(ccr, 4),
        "TotalScore": round(total, 4),
    }


def dialogue_to_text(dialogue: List[Dict]) -> str:
    return "\n".join([f"{t.get('role', '')}: {t.get('content', '')}" for t in dialogue])


def build_objective_dpo_pairs(
    results: Dict[str, List],
    topic_idx: int,
    topic_text: str,
    chosen_name: str = "socratic_chosen",
    min_obj_total: float = 0.72,
    min_obj_margin: float = 0.08,
    min_objective_wins: int = 4,
) -> List[Dict[str, Any]]:
    """基于客观分门控构建 DPO pair，优先保证 objective_eval 可提升。"""
    chosen_runs = results.get(chosen_name, [])
    rejected_names = [name for name in results.keys() if name.endswith("_rejected")]
    objective_keys = [
        "StrategyDensity",
        "StrategyVariety",
        "IKT",
        "BloomProgression",
        "StructureCompleteness",
        "L3GuidanceRate",
        "CognitiveCorrectionRate",
    ]

    dpo_pairs: List[Dict[str, Any]] = []
    for c_idx, chosen in enumerate(chosen_runs):
        c_scores = chosen.get("scores", {})
        c_total = float(c_scores.get("TotalScore", 0.0))
        if c_total < min_obj_total:
            continue

        for rejected_name in rejected_names:
            for r_idx, rejected in enumerate(results.get(rejected_name, [])):
                r_scores = rejected.get("scores", {})
                r_total = float(r_scores.get("TotalScore", 0.0))

                margin = c_total - r_total
                if margin < min_obj_margin:
                    continue

                objective_wins = 0
                for k in objective_keys:
                    if float(c_scores.get(k, 0.0)) > float(r_scores.get(k, 0.0)):
                        objective_wins += 1
                if objective_wins < min_objective_wins:
                    continue

                pair = {
                    "id": f"topic{topic_idx}_c{c_idx}_r{rejected_name}_{r_idx}",
                    "topic_idx": topic_idx,
                    "topic": topic_text,
                    "prompt": topic_text,
                    "chosen": dialogue_to_text(chosen.get("dialogue", [])),
                    "rejected": dialogue_to_text(rejected.get("dialogue", [])),
                    "chosen_prompt_name": chosen_name,
                    "rejected_prompt_name": rejected_name,
                    "chosen_scores": c_scores,
                    "rejected_scores": r_scores,
                    "objective_margin": round(margin, 4),
                    "objective_wins": objective_wins,
                }
                dpo_pairs.append(pair)

    return dpo_pairs


def score_subjective(dialogue: List[Dict], topic_text: str) -> Dict[str, float]:
    """对对话进行主观评测 (复用 subjective_eval 逻辑)"""
    if not dialogue:
        return {}

    dialogue_text = "\n".join([f"{t['role']}: {t['content']}" for t in dialogue])

    prompt = f"""
你是一位教育专家，正在评估以下教师与学生之间的跨学科对话质量。请根据五个关键指标进行评测：
=== 对话场景 ===
学科主题/课程内容: {topic_text}
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
"""

    result = call_llm(
        [{"role": "user", "content": prompt}],
        model=ANNOTATOR_MODEL,
        temperature=0.1,
        max_tokens=1024,
    )
    if not result:
        return {}

    parsed = {}
    try:
        import re

        no_code_blocks = re.sub(
            r"^```(?:json)?|```$", "", result.strip(), flags=re.IGNORECASE
        ).strip()
        json_match = re.search(r"\{[\s\S]*\}", no_code_blocks)
        if json_match:
            parsed = json.loads(json_match.group(0))
    except Exception as e:
        logger.error(f"Subjective eval JSON error: {e}")
        return {}

    scores = {}
    for k in ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]:
        val = parsed.get(k, {}).get("score", 0)
        scores[k] = float(val) if val else 0.0

    return scores


# ── 主流程 ──
def run_test(
    topic_text: str,
    prompt_names: List[str],
    repeats: int = 3,
    objective_only: bool = False,
    max_turns: int = 8,
):
    """对每个 prompt 生成 repeats 组对话，标注打分，输出对比"""
    results = {name: [] for name in prompt_names}

    total_runs = len(prompt_names) * repeats
    current = 0

    student_types = [
        "你是一位基础薄弱的学生，对概念掌握不牢",
        "你是一位有一定基础的学生，善于类比推理",
        "你是一位全优型学生，逻辑清晰",
    ]
    scenarios = [
        "学生尝试回答但有部分错误",
        "学生进行了一个类比猜测",
        "学生回答正确并尝试延伸",
        "学生不太理解，请求老师解释",
    ]

    for r in range(repeats):
        # 每个 round 只采样一次学生轨迹，并对所有 prompt 复用，保证横向可比
        init_prompt = f"""你是一名中学生，正在学习一节跨学科课程。请根据下面的课程内容，提出一个你真正感到困惑或好奇的问题。
要求：回答要口语化，符合中学生身份。只输出问题内容。

课程内容如下：
{topic_text}"""
        shared_initial_question = call_llm(
            [{"role": "user", "content": init_prompt}], temperature=0.9
        )
        if not shared_initial_question:
            print(f"\n[Round {r + 1}] ❌ 无法生成共享学生初始问题，跳过该轮")
            continue

        shared_student_plan = [
            {
                "identity": random.choice(student_types),
                "scenario": random.choice(scenarios),
            }
            for _ in range(max_turns)
        ]

        for name in prompt_names:
            prompt = PROMPTS[name]
            if r == 0:
                print(f"\n{'=' * 60}")
                print(f"Testing Prompt: [{name}]")
                print(f"{'=' * 60}")

            current += 1
            print(
                f"  [{current}/{total_runs}] {name} - Round {r + 1}...",
                end=" ",
                flush=True,
            )

            # 1. 生成对话
            dialogue = generate_dialogue(
                topic_text,
                prompt,
                max_turns=max_turns,
                prompt_name=name,
                initial_question=shared_initial_question,
                student_plan=shared_student_plan,
            )
            if not dialogue or len(dialogue) < 4:
                print(f"❌ 对话太短({len(dialogue)}轮)，跳过")
                # 记录一下到底说了啥
                if dialogue:
                    logger.info(f"Skipped dialogue (name={name}): {dialogue}")
                continue

            print(f"({len(dialogue)}轮)", end=" ", flush=True)

            # 2. 标注
            annotations = annotate_dialogue(dialogue)
            if not annotations:
                print("❌ 标注失败")
                continue

            # 3. 客观打分
            obj_scores = score_annotations(annotations)
            if not obj_scores:
                print("❌ 客观打分失败")
                continue

            # 4. 主观打分（objective_only 模式可跳过）
            subj_scores = {}
            if not objective_only:
                subj_scores = score_subjective(dialogue, topic_text)

            if objective_only:
                scores = {**obj_scores}
            else:
                # 5. 表面质量打分
                surface_scores = score_dialogue_surface(dialogue)

                avg_subj = (
                    sum(
                        subj_scores.get(k, 0.0)
                        for k in ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
                    )
                    / 5.0
                )
                dpo_readiness = (
                    0.50 * obj_scores.get("TotalScore", 0.0)
                    + 0.40 * (avg_subj / 5.0)
                    + 0.10 * surface_scores.get("QuestionRate", 0.0)
                    - 0.10 * surface_scores.get("FillerPenalty", 0.0)
                )
                dpo_readiness = clamp01(dpo_readiness)

                # 合并分数
                scores = {
                    **obj_scores,
                    **subj_scores,
                    **surface_scores,
                    "DPOReadiness": round(dpo_readiness, 4),
                }

            results[name].append({
                "scores": scores,
                "dialogue_turns": len(dialogue),
                "dialogue": dialogue,
                "annotations": annotations,
            })
            if objective_only:
                print(
                    f"✅ ObjTotal={obj_scores.get('TotalScore', 0):.4f} "
                    f"(objective-only mode)"
                )
            else:
                avg_subj = (
                    sum(
                        subj_scores.get(k, 0.0)
                        for k in ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
                    )
                    / 5.0
                )
                print(
                    f"✅ ObjTotal={obj_scores.get('TotalScore', 0):.4f} "
                    f"SubjAvg={avg_subj:.2f} DPO={scores.get('DPOReadiness', 0):.3f}"
                )

    return results


def summarize_results(results: Dict[str, List]) -> Dict[str, Dict[str, float]]:
    """汇总每个 prompt 的平均分。"""
    obj_metrics = ["TotalScore", "IKT", "L3GuidanceRate", "BloomProgression"]
    subj_metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    ext_metrics = ["QuestionRate", "FillerPenalty", "DPOReadiness"]
    all_metrics = obj_metrics + subj_metrics + ext_metrics

    summaries: Dict[str, Dict[str, float]] = {}
    for name, runs in results.items():
        if not runs:
            continue
        avgs: Dict[str, float] = {}
        for m in all_metrics:
            values = [r["scores"].get(m, 0.0) for r in runs]
            avgs[m] = (sum(values) / len(values)) if values else 0.0
        avgs["N"] = float(len(runs))
        avgs["SubjAvg"] = sum(avgs[sm] for sm in subj_metrics) / len(subj_metrics)
        summaries[name] = avgs
    return summaries


def check_dpo_ready(
    summaries: Dict[str, Dict[str, float]],
    chosen_name: str = "socratic_chosen",
    min_runs: int = 3,
    min_chosen_obj: float = 0.72,
    min_chosen_subj: float = 4.20,
    min_margin_obj: float = 0.08,
    min_margin_subj: float = 1.20,
    min_chosen_dpo: float = 0.78,
) -> Dict[str, Any]:
    """判断是否达到可用于 DPO 的 prompt 区分度。"""
    if chosen_name not in summaries:
        return {
            "ready": False,
            "reason": f"缺少 {chosen_name} 的有效结果",
        }

    chosen = summaries[chosen_name]
    rejected_names = [k for k in summaries.keys() if k.endswith("_rejected")]
    if not rejected_names:
        return {
            "ready": False,
            "reason": "缺少 rejected prompt 结果，无法比较",
        }

    n_ok = int(chosen.get("N", 0)) >= min_runs
    chosen_obj = chosen.get("TotalScore", 0.0)
    chosen_subj = chosen.get("SubjAvg", 0.0)
    chosen_dpo = chosen.get("DPOReadiness", 0.0)

    best_rejected_obj = max(summaries[r].get("TotalScore", 0.0) for r in rejected_names)
    best_rejected_subj = max(summaries[r].get("SubjAvg", 0.0) for r in rejected_names)

    obj_margin = chosen_obj - best_rejected_obj
    subj_margin = chosen_subj - best_rejected_subj

    ready = all([
        n_ok,
        chosen_obj >= min_chosen_obj,
        chosen_subj >= min_chosen_subj,
        obj_margin >= min_margin_obj,
        subj_margin >= min_margin_subj,
        chosen_dpo >= min_chosen_dpo,
    ])

    return {
        "ready": ready,
        "n_ok": n_ok,
        "chosen_obj": round(chosen_obj, 4),
        "chosen_subj": round(chosen_subj, 4),
        "chosen_dpo": round(chosen_dpo, 4),
        "best_rejected_obj": round(best_rejected_obj, 4),
        "best_rejected_subj": round(best_rejected_subj, 4),
        "obj_margin": round(obj_margin, 4),
        "subj_margin": round(subj_margin, 4),
        "reason": "达标" if ready else "未达到 DPO 可用阈值",
    }


def print_summary(results: Dict[str, List], objective_only: bool = False):
    """打印对比汇总表"""
    print(f"\n{'=' * 110}")
    if objective_only:
        print("📊 PROMPT A/B TEST SUMMARY (Objective Only)")
    else:
        print("📊 PROMPT A/B TEST SUMMARY (Objective + Subjective)")
    print(f"{'=' * 110}")

    obj_metrics = ["TotalScore", "IKT", "L3GuidanceRate", "BloomProgression"]
    subj_metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    if objective_only:
        ext_metrics = ["StrategyV", "Structur", "N"]
        all_metrics = obj_metrics + ext_metrics
    else:
        ext_metrics = ["QuestionRate", "DPORead", "N"]
        all_metrics = obj_metrics + subj_metrics + ext_metrics

    # 表头
    header = f"{'Prompt':<25}"
    for m in all_metrics:
        header += f" {m[:8]:>8}"
    header += f" {'N':>3}"
    print(header)
    print("-" * len(header))

    summaries = {}
    summaries = summarize_results(results)
    for name, avgs in summaries.items():
        runs = results.get(name, [])
        if not runs:
            print(f"{name:<25}  -- no valid runs --")
            continue

        row = f"{name:<25}"
        for m in obj_metrics:
            row += f" {avgs[m]:>8.3f}"
        if objective_only:
            row += f" {avgs['StrategyVariety']:>8.3f}"
            row += f" {avgs['StructureCompleteness']:>8.3f}"
        else:
            for m in subj_metrics:
                row += f" {avgs[m]:>8.2f}"
            row += f" {avgs['QuestionRate']:>8.2f}"
            row += f" {avgs['DPOReadiness']:>8.3f}"
        row += f" {len(runs):>3}"
        print(row)

    # 找主客观综合表现 (Obj Total + Avg Subjective / 5)
    if summaries:
        print(f"\n{'─' * 110}")
        for name, avgs in summaries.items():
            if objective_only:
                print(
                    f"- {name}: Obj={avgs['TotalScore']:.3f}, IKT={avgs['IKT']:.3f}, "
                    f"SC={avgs['StructureCompleteness']:.3f}, L3={avgs['L3GuidanceRate']:.3f}"
                )
            else:
                avg_subj = avgs["SubjAvg"]
                print(
                    f"- {name}: Obj={avgs['TotalScore']:.3f}, Subj(Avg)={avg_subj:.2f}/5.0, "
                    f"DPO={avgs['DPOReadiness']:.3f}"
                )


def run_until_dpo_ready(
    topic_text: str,
    prompt_names: List[str],
    repeats: int,
    max_iterations: int,
    min_runs: int,
    min_chosen_obj: float,
    min_chosen_subj: float,
    min_margin_obj: float,
    min_margin_subj: float,
    min_chosen_dpo: float,
) -> Dict[str, Any]:
    """自动重复测试直到达到 DPO 可用阈值或达到最大轮次。"""
    history: List[Dict[str, Any]] = []
    latest_results: Dict[str, List] = {}
    latest_gate: Dict[str, Any] = {"ready": False, "reason": "未开始"}

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'#' * 60}")
        print(f"🔁 Iteration {iteration}/{max_iterations}")
        print(f"{'#' * 60}")

        latest_results = run_test(topic_text, prompt_names, repeats)
        print_summary(latest_results)

        summaries = summarize_results(latest_results)
        latest_gate = check_dpo_ready(
            summaries,
            chosen_name="socratic_chosen",
            min_runs=min_runs,
            min_chosen_obj=min_chosen_obj,
            min_chosen_subj=min_chosen_subj,
            min_margin_obj=min_margin_obj,
            min_margin_subj=min_margin_subj,
            min_chosen_dpo=min_chosen_dpo,
        )
        history.append({
            "iteration": iteration,
            "summary": summaries,
            "gate": latest_gate,
        })

        print("\n📌 DPO Gate:")
        print(json.dumps(latest_gate, ensure_ascii=False, indent=2))

        if latest_gate.get("ready"):
            print("✅ 已达到 DPO 训练可用阈值，停止迭代。")
            break

    return {
        "results": latest_results,
        "history": history,
        "gate": latest_gate,
    }


def parse_topic_indices(raw: str, topics_count: int) -> List[int]:
    if not raw.strip():
        return [0, 1, 2] if topics_count >= 3 else list(range(topics_count))
    indices: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        val = int(piece)
        if val < 0 or val >= topics_count:
            raise ValueError(f"topic index {val} out of range [0, {topics_count - 1}]")
        indices.append(val)
    return sorted(set(indices))


def select_best_prompt_by_objective(
    topics: List[Dict[str, Any]],
    topic_indices: List[int],
    prompt_names: List[str],
    repeats: int,
    max_turns: int,
) -> Dict[str, Any]:
    """小规模多 topic 测试，按 objective 稳定性选择最佳 prompt。"""
    all_scores: Dict[str, List[float]] = {name: [] for name in prompt_names}
    per_topic_summary: Dict[str, Dict[str, float]] = {}

    for idx in topic_indices:
        topic_text = str(topics[idx].get("topic", ""))
        print(f"\n{'#' * 72}")
        print(f"🔎 Small-Scale Objective Selection | Topic #{idx}")
        print(f"{'#' * 72}")

        results = run_test(
            topic_text,
            prompt_names,
            repeats=repeats,
            objective_only=True,
            max_turns=max_turns,
        )
        summary = summarize_results(results)
        per_topic_summary[str(idx)] = {
            k: v.get("TotalScore", 0.0) for k, v in summary.items()
        }

        for name in prompt_names:
            runs = results.get(name, [])
            for run in runs:
                all_scores[name].append(
                    float(run.get("scores", {}).get("TotalScore", 0.0))
                )

    ranking_rows: List[Dict[str, Any]] = []
    for name in prompt_names:
        vals = all_scores.get(name, [])
        if not vals:
            ranking_rows.append({
                "prompt": name,
                "n": 0,
                "obj_mean": 0.0,
                "obj_std": 1.0,
                "obj_min": 0.0,
                "stable_score": -1.0,
            })
            continue

        obj_mean = statistics.mean(vals)
        obj_std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        obj_min = min(vals)
        # 稳定性优先：均值高 + 波动小 + 最差样本不塌
        stable_score = obj_mean - 0.5 * obj_std + 0.2 * obj_min
        ranking_rows.append({
            "prompt": name,
            "n": len(vals),
            "obj_mean": round(obj_mean, 4),
            "obj_std": round(obj_std, 4),
            "obj_min": round(obj_min, 4),
            "stable_score": round(stable_score, 4),
        })

    ranking_rows.sort(key=lambda x: x["stable_score"], reverse=True)
    winner = ranking_rows[0]["prompt"] if ranking_rows else ""

    print(f"\n{'=' * 88}")
    print("🏁 BEST PROMPT SELECTION (Objective-Focused)")
    print(f"{'=' * 88}")
    print(
        f"{'Prompt':<28} {'N':>4} {'ObjMean':>10} {'ObjStd':>10} {'ObjMin':>10} {'Stable':>10}"
    )
    print("-" * 88)
    for row in ranking_rows:
        print(
            f"{row['prompt']:<28} {row['n']:>4} {row['obj_mean']:>10.4f} "
            f"{row['obj_std']:>10.4f} {row['obj_min']:>10.4f} {row['stable_score']:>10.4f}"
        )
    print(f"\n✅ 推荐最佳 prompt: {winner}")

    return {
        "winner": winner,
        "topic_indices": topic_indices,
        "ranking": ranking_rows,
        "per_topic_objective": per_topic_summary,
    }


def run_quick_iteration(
    topics: List[Dict[str, Any]],
    prompt_names: List[str],
    max_turns: int,
    output_path: str,
) -> Dict[str, Any]:
    """快速单轮迭代：同一随机 topic 下同时跑多个 prompt，直接比较 objective 分差。"""
    if not prompt_names:
        raise ValueError("prompt_names is empty")
    for name in prompt_names:
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt name: {name}")
    if not topics:
        raise ValueError("No topics loaded")

    topic_idx = random.randrange(len(topics))
    topic_text = str(topics[topic_idx].get("topic", ""))

    print(f"\n{'#' * 72}")
    print("⚡ QUICK ITERATION MODE")
    print(f"- Prompts: {', '.join(prompt_names)}")
    print(f"- Random Topic Index: {topic_idx}")
    print(f"{'#' * 72}")

    results = run_test(
        topic_text=topic_text,
        prompt_names=prompt_names,
        repeats=1,
        objective_only=True,
        max_turns=max(4, max_turns),
    )

    summary = summarize_results(results)
    per_prompt: Dict[str, Any] = {}
    for name in prompt_names:
        runs = results.get(name, [])
        if not runs:
            per_prompt[name] = {"ok": False, "reason": "no_valid_runs"}
            continue
        run = runs[0]
        teacher_turns = [
            t.get("content", "")
            for t in run.get("dialogue", [])
            if t.get("role") == "教师"
        ]
        per_prompt[name] = {
            "ok": True,
            "scores": run.get("scores", {}),
            "teacher_preview": teacher_turns[:2],
        }

    chosen_total = float(summary.get("socratic_chosen", {}).get("TotalScore", 0.0))
    rejected_candidates = [
        float(summary.get(name, {}).get("TotalScore", 0.0))
        for name in prompt_names
        if name.endswith("_rejected")
    ]
    best_rejected_total = max(rejected_candidates) if rejected_candidates else 0.0
    objective_gap = chosen_total - best_rejected_total

    payload = {
        "ok": bool(summary),
        "topic_idx": topic_idx,
        "topic": topic_text,
        "prompt_names": prompt_names,
        "objective_summary": {
            name: {
                "TotalScore": round(
                    float(summary.get(name, {}).get("TotalScore", 0.0)), 4
                ),
                "IKT": round(float(summary.get(name, {}).get("IKT", 0.0)), 4),
                "StructureCompleteness": round(
                    float(summary.get(name, {}).get("StructureCompleteness", 0.0)), 4
                ),
                "L3GuidanceRate": round(
                    float(summary.get(name, {}).get("L3GuidanceRate", 0.0)), 4
                ),
                "BloomProgression": round(
                    float(summary.get(name, {}).get("BloomProgression", 0.0)), 4
                ),
                "StrategyVariety": round(
                    float(summary.get(name, {}).get("StrategyVariety", 0.0)), 4
                ),
            }
            for name in prompt_names
        },
        "objective_gap": round(objective_gap, 4),
        "chosen_total": round(chosen_total, 4),
        "best_rejected_total": round(best_rejected_total, 4),
        "per_prompt": per_prompt,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    history_path = Path(output_path).with_suffix(".history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"\n💾 Quick iteration result saved to {output_path}")
    print(f"🧾 Quick iteration history appended to {history_path}")

    if payload.get("ok"):
        print("📌 Objective Snapshot (single random topic)")
        for name in prompt_names:
            s = payload["objective_summary"].get(name, {})
            print(
                f"- {name}: Total={s.get('TotalScore', 0):.4f}, IKT={s.get('IKT', 0):.4f}, "
                f"SC={s.get('StructureCompleteness', 0):.4f}, L3={s.get('L3GuidanceRate', 0):.4f}, "
                f"BP={s.get('BloomProgression', 0):.4f}, SV={s.get('StrategyVariety', 0):.4f}"
            )
        print(
            f"📌 Gap(chosen-best_rejected): {payload.get('chosen_total', 0):.4f} - "
            f"{payload.get('best_rejected_total', 0):.4f} = {payload.get('objective_gap', 0):.4f}"
        )
    else:
        print("❌ Quick iteration produced no valid run.")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Prompt A/B Test with QwenFlash")
    parser.add_argument(
        "--topic-idx", type=int, default=1, help="Topic index (0-based)"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per prompt")
    parser.add_argument(
        "--prompts",
        type=str,
        default="all",
        help="Comma-separated prompt names, or 'all'",
    )
    parser.add_argument(
        "--output", type=str, default="prompt_ab_results.json", help="Output JSON"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max repeat iterations until DPO-ready",
    )
    parser.add_argument(
        "--min-runs", type=int, default=3, help="Min valid runs for chosen prompt"
    )
    parser.add_argument(
        "--min-chosen-obj", type=float, default=0.72, help="Chosen min objective total"
    )
    parser.add_argument(
        "--min-chosen-subj",
        type=float,
        default=4.2,
        help="Chosen min subjective avg (1~5)",
    )
    parser.add_argument(
        "--min-margin-obj",
        type=float,
        default=0.08,
        help="Chosen - best rejected objective margin",
    )
    parser.add_argument(
        "--min-margin-subj",
        type=float,
        default=1.2,
        help="Chosen - best rejected subjective margin",
    )
    parser.add_argument(
        "--min-chosen-dpo", type=float, default=0.78, help="Chosen min DPO readiness"
    )
    parser.add_argument(
        "--export-dpo-objective",
        action="store_true",
        help="Export objective-gated DPO pairs",
    )
    parser.add_argument(
        "--dpo-output",
        type=str,
        default="dpo_objective_pairs.jsonl",
        help="Output JSONL for DPO pairs",
    )
    parser.add_argument(
        "--dpo-min-obj-total",
        type=float,
        default=0.72,
        help="Chosen min objective total for DPO pair",
    )
    parser.add_argument(
        "--dpo-min-obj-margin",
        type=float,
        default=0.08,
        help="Chosen-Rejected min objective margin for DPO pair",
    )
    parser.add_argument(
        "--dpo-min-objective-wins",
        type=int,
        default=4,
        help="Minimum objective metric wins in chosen over rejected",
    )
    parser.add_argument(
        "--select-best-prompt",
        action="store_true",
        help="Run small-scale multi-topic objective selection",
    )
    parser.add_argument(
        "--topic-indices",
        type=str,
        default="",
        help="Comma-separated topic indices for selection, e.g. 0,1,2",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Max teacher turns per generated dialogue (支持展现策略多样性与SV)",
    )
    parser.add_argument(
        "--objective-only",
        action="store_true",
        help="Skip subjective scoring and only compute objective metrics",
    )
    parser.add_argument(
        "--quick-iterate",
        action="store_true",
        help="Fast loop: random topic + single run + objective snapshot",
    )
    args = parser.parse_args()

    # 加载 topic
    topic_file = (
        Path(__file__).parent.parent.parent / "data" / "interdisciplinary_topic.json"
    )
    with open(topic_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if args.topic_idx >= len(topics):
        print(f"Error: topic-idx {args.topic_idx} out of range (max {len(topics) - 1})")
        sys.exit(1)

    topic = topics[args.topic_idx]
    topic_text = topic.get("topic", "")
    print(f"📚 Topic #{args.topic_idx}: {topic_text[:80]}...")

    # 选择要测试的 prompt
    if args.prompts == "all":
        prompt_names = list(PROMPTS.keys())
    else:
        prompt_names = [p.strip() for p in args.prompts.split(",")]

    print(
        f"🧪 Testing {len(prompt_names)} prompts × {args.repeats} repeats = {len(prompt_names) * args.repeats} runs"
    )

    if args.quick_iterate:
        run_quick_iteration(
            topics=topics,
            prompt_names=prompt_names,
            max_turns=max(4, args.max_turns),
            output_path=args.output,
        )
        return

    if args.select_best_prompt:
        try:
            topic_indices = parse_topic_indices(args.topic_indices, len(topics))
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        selection = select_best_prompt_by_objective(
            topics=topics,
            topic_indices=topic_indices,
            prompt_names=prompt_names,
            repeats=args.repeats,
            max_turns=max(4, args.max_turns),
        )
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selection, f, ensure_ascii=False, indent=2)
        print(f"\n💾 最佳 prompt 选择结果已保存到 {output_path}")
        return

    # 单次运行：按当前 prompt 配置执行 A/B 测试
    results = run_test(
        topic_text,
        prompt_names,
        args.repeats,
        objective_only=args.objective_only,
        max_turns=max(4, args.max_turns),
    )

    # 输出汇总
    print_summary(results, objective_only=args.objective_only)

    # 保存详细结果
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存到 {output_path}")

    if args.export_dpo_objective:
        dpo_pairs = build_objective_dpo_pairs(
            results=results,
            topic_idx=args.topic_idx,
            topic_text=topic_text,
            chosen_name="socratic_chosen",
            min_obj_total=args.dpo_min_obj_total,
            min_obj_margin=args.dpo_min_obj_margin,
            min_objective_wins=max(1, args.dpo_min_objective_wins),
        )
        dpo_output = Path(args.dpo_output)
        with open(dpo_output, "w", encoding="utf-8") as f:
            for item in dpo_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"🧾 Objective-gated DPO pairs: {len(dpo_pairs)} 条，已保存到 {dpo_output}"
        )


if __name__ == "__main__":
    main()
