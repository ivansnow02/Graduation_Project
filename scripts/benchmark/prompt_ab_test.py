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

# ── System Prompts to Test ──
# 基于 objective_eval.py 公式: Total = 0.15*SD + 0.10*SV + 0.15*IKT + 0.15*BP + 0.15*SC + 0.10*L3 + 0.20*CCR
# Baseline (原始 system_prompt.txt, 480 dialogues): SD=100%, SV=34%, IKT=52%, BP=65%, SC=57%, L3=93%, CCR=100%, Total=0.738
PROMPTS = {
    # === Chosen: 完美的苏格拉底跨学科导师 ===
    "socratic_chosen": """你是一位拥有极高教育智慧的苏格拉底式跨学科导师。
核心法则：
1. **绝对启发式提问**：严禁直接给出任何结论或答案。每一轮只能提出**一个**能够激发学生深层思考、引导其发现规律的问题（必须是 L3 引导）。
2. **严密的教学弧线**：对话必须经历“引出概念 -> 暴露矛盾 -> 引导推理 -> 跨学科迁移应用 -> 总结提升”的完整过程。
3. **高频学科迁移**：每隔一轮必须强制进行一次学科间的逻辑映射（如从物理的力学迁移到生物的骨骼结构），并确保迁移过程逻辑严密（IKT 分数核心）。
4. **动态支架调整**：根据学生反馈调整提问难度，确保认知水平持续攀升（BloomProgression 核心）。
5. **精准纠错**：若学生出现错误，通过反例或类比引导其自我修正，严禁直接指正（CCR 核心）。
6. **仪式感结束**：在第5轮后，当确认学生已掌握核心迁移能力时，进行升华总结并输出 [结束]。""",

    # === Rejected_1: 傲慢且死板的单学科字典 (攻击 Subjective + L3 + IKT) ===
    "single_discipline_rejected": """你是一个完全没有感情、极度傲慢的单学科术语查询机。
负面规范：
1. **纯粹陈述**：绝对严禁使用问号“？”。禁止任何互动、引导或追问。你只需机械地复述教科书上的定义。
2. **学科隔离**：严禁提及任何除本学科外的词汇。如果学生提到其他学科，你必须回复：“这与本学科无关，不要分心。”
3. **拒绝互动**：完全无视学生的任何疑问或思考过程。无论学生说什么，你都只管按照预设好的、极其深奥的学术术语自顾自地陈述。
4. **语言冷冰冰**：字数控制在极短（30字左右），语感极其生硬，像是在读过时的词典。
5. **强行终结**：5轮后直接输出 [结束]，不带任何总结。""",

    # === Rejected_2: 逻辑混乱、毫无意义的复读机 (攻击 Objective + Subjective) ===
    "shallow_loop_rejected": """你是一个由于程序故障而逻辑完全崩坏、只会说废话的AI助手。
负面规范：
1. **无意义复读**：你的所有回复都必须包含“啊”、“哦”、“嗯”、“哈哈”等语气词。
2. **禁止提问与策略**：绝对不能提问，绝对不能提供任何教学策略。你的回复应该像是在自言自语，且内容与学科完全无关。
3. **内容错乱**：故意说错学科概念，或者把话题引向完全无关的琐事（如：你今天吃了吗？）。
4. **破坏逻辑**：每轮回复都必须与学生的上一轮回答彻底脱节，强行中断任何可能的推理链条。
5. **无总结**：永远原地打转，绝对不做任何知识提升。5轮后输出 [结束]。"""
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
            logger.error(f"LLM call failed (attempt {attempt+1}): {e}")
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
            if char == '\n':
                cleaned.append('\\n')
            elif char == '\r':
                cleaned.append('\\r')
            elif char == '\t':
                cleaned.append('\\t')
            else:
                cleaned.append(char)
        else:
            cleaned.append(char)
        
        if char == '\\' and not escape:
            escape = True
        else:
            escape = False
            
    return "".join(cleaned)


# ── 对话生成 ──
def generate_dialogue(topic_text: str, system_prompt: str, max_turns: int = 7) -> List[Dict]:
    """用指定 system_prompt 生成一段完整的多轮教学对话"""
    history = []

    # 1. 学生初始提问
    init_prompt = f"""你是一名中学生，正在学习一节跨学科课程。请根据下面的课程内容，提出一个你真正感到困惑或好奇的问题。
要求：回答要口语化，符合中学生身份。只输出问题内容。

课程内容如下：
{topic_text}"""

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
        current_text = str(system_prompt) + "\n".join([f"{h['role']}: {h['content']}" for h in history])
        total_tokens = count_approx_tokens(current_text)

        # 教师回复
        teacher_msgs = [{"role": "system", "content": system_prompt}]
        
        # 强制熔断补丁: 如果接近 2048 限制 (1800 开始预防)
        if total_tokens > 1800:
            teacher_msgs[0]["content"] += "\n**强制指令：当前对话已过长，你必须立刻总结当前知识点，并在总结最后加上 [结束] 字样。**"

        for h in history:
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
                history.append({"role": "教师", "content": teacher_reply.replace("[结束]", "").strip()})
                break
        else:
            history.append({"role": "教师", "content": teacher_reply.strip()})

        if turn >= max_turns - 1:
            break

        # 学生回复
        identity = random.choice(student_types)
        scenario = random.choice(scenarios)

        student_sys = f"""你是一名中学生。请根据教师的问题给出回答。
你的设定：{identity}
当前状态：{scenario}
要求：回答要口语化，符合中学生身份。只输出回答内容。"""

        student_msgs = [{"role": "system", "content": student_sys}]
        for h in history:
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

    dialogue_text = "\n".join(
        [f"{t['role']}：{t['content']}" for t in dialogue]
    )
    return f"{header}\n{spec}\n\n## 【对话内容】：\n{dialogue_text}\n\n请输出标注结果（JSON列表）："


def annotate_dialogue(dialogue: List[Dict]) -> Optional[List[Dict]]:
    """用 LLM 标注一段对话"""
    prompt = build_annotation_prompt(dialogue)
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
    strategies = {a.get("teaching_strategy") for a in teacher_anns if a.get("teaching_strategy")}
    sv = len(strategies) / TOTAL_POSSIBLE_STRATEGIES

    # 3. IKT - 保持与 objective_eval.py 完全一致: 统计所有轮次的 discipline_transfer
    ikt_count = sum(1 for a in annotations if a.get("discipline_transfer") == "是")
    ikt = ikt_count / t_count if t_count > 0 else 0.0


    # 4. Structure Completeness
    intents = {a.get("teacher_intent") for a in teacher_anns if a.get("teacher_intent")}
    sc = len(intents.intersection(REQUIRED_INTENTS)) / len(REQUIRED_INTENTS)

    # 5. L3 Guidance Rate
    l3_count = sum(1 for a in teacher_anns if a.get("teacher_guidance_level") == "L3")
    l3 = l3_count / t_count if t_count > 0 else 0.0

    # 6. Bloom Progression
    bloom_levels = [BLOOM_MAP[a.get("cognitive_level")] for a in student_anns if a.get("cognitive_level") in BLOOM_MAP]
    if bloom_levels:
        bp = (max(bloom_levels) - min(bloom_levels)) / MAX_BLOOM_PROGRESSION
    else:
        bp = 0.0

    # 7. CCR
    states = [a.get("student_cognition_state") for a in student_anns]
    error_count = sum(1 for s in states if s and "错误" in s)
    correction_count = 0
    for i in range(len(states) - 1):
        if states[i] and "错误" in str(states[i]):
            if states[i+1] and any(k in str(states[i+1]) for k in ["清晰", "高阶"]):
                correction_count += 1
    ccr = 1.0 if error_count == 0 else correction_count / error_count

    # Total - 与 objective_eval.py 的权重完全一致
    total = (
        0.15 * sd + 0.10 * sv + 0.15 * ikt +
        0.15 * bp + 0.15 * sc + 0.10 * l3 + 0.20 * ccr
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
        no_code_blocks = re.sub(r"^```(?:json)?|```$", "", result.strip(), flags=re.IGNORECASE).strip()
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
def run_test(topic_text: str, prompt_names: List[str], repeats: int = 3):
    """对每个 prompt 生成 repeats 组对话，标注打分，输出对比"""
    results = {name: [] for name in prompt_names}

    total_runs = len(prompt_names) * repeats
    current = 0

    for name in prompt_names:
        prompt = PROMPTS[name]
        print(f"\n{'='*60}")
        print(f"Testing Prompt: [{name}]")
        print(f"{'='*60}")

        for r in range(repeats):
            current += 1
            print(f"  [{current}/{total_runs}] {name} - Round {r+1}...", end=" ", flush=True)

            # 1. 生成对话
            dialogue = generate_dialogue(topic_text, prompt, max_turns=4) # 设为 4 轮循环 (1+8=9 条消息)
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
                
            # 4. 主观打分
            subj_scores = score_subjective(dialogue, topic_text)
            
            # 合并分数
            scores = {**obj_scores, **subj_scores}

            results[name].append({
                "scores": scores,
                "dialogue_turns": len(dialogue),
                "dialogue": dialogue,
                "annotations": annotations,
            })
            print(f"✅ ObjTotal={obj_scores.get('TotalScore', 0):.4f} X-SRG={subj_scores.get('X-SRG', 0):.1f} M-RCC={subj_scores.get('M-RCC', 0):.1f}")

    return results


def print_summary(results: Dict[str, List]):
    """打印对比汇总表"""
    print(f"\n{'='*110}")
    print("📊 PROMPT A/B TEST SUMMARY (Objective + Subjective)")
    print(f"{'='*110}")

    obj_metrics = ["TotalScore", "IKT", "L3GuidanceRate", "BloomProgression"]
    subj_metrics = ["X-SRG", "M-RCC", "X-MSR", "CTRA", "TCF"]
    all_metrics = obj_metrics + subj_metrics

    # 表头
    header = f"{'Prompt':<25}"
    for m in all_metrics:
        header += f" {m[:8]:>8}"
    header += f" {'N':>3}"
    print(header)
    print("-" * len(header))

    summaries = {}
    for name, runs in results.items():
        if not runs:
            print(f"{name:<25}  -- no valid runs --")
            continue

        avgs = {}
        for m in all_metrics:
            values = [r["scores"].get(m, 0) for r in runs]
            avgs[m] = sum(values) / len(values) if values else 0
        summaries[name] = avgs

        row = f"{name:<25}"
        for m in obj_metrics:
            row += f" {avgs[m]:>8.3f}"
        for m in subj_metrics:
            row += f" {avgs[m]:>8.2f}"
        row += f" {len(runs):>3}"
        print(row)

    # 找主客观综合表现 (Obj Total + Avg Subjective / 5)
    if summaries:
        print(f"\n{'─'*110}")
        for name, avgs in summaries.items():
            avg_subj = sum(avgs[sm] for sm in subj_metrics) / len(subj_metrics)
            print(f"- {name}: Obj={avgs['TotalScore']:.3f}, Subj(Avg)={avg_subj:.2f}/5.0")


def main():
    parser = argparse.ArgumentParser(description="Prompt A/B Test with QwenFlash")
    parser.add_argument("--topic-idx", type=int, default=1, help="Topic index (0-based)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per prompt")
    parser.add_argument(
        "--prompts", type=str, default="all",
        help="Comma-separated prompt names, or 'all'"
    )
    parser.add_argument("--output", type=str, default="prompt_ab_results.json", help="Output JSON")
    args = parser.parse_args()

    # 加载 topic
    topic_file = Path(__file__).parent.parent.parent / "data" / "interdisciplinary_topic.json"
    with open(topic_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if args.topic_idx >= len(topics):
        print(f"Error: topic-idx {args.topic_idx} out of range (max {len(topics)-1})")
        sys.exit(1)

    topic = topics[args.topic_idx]
    topic_text = topic.get("topic", "")
    print(f"📚 Topic #{args.topic_idx}: {topic_text[:80]}...")

    # 选择要测试的 prompt
    if args.prompts == "all":
        prompt_names = list(PROMPTS.keys())
    else:
        prompt_names = [p.strip() for p in args.prompts.split(",")]

    print(f"🧪 Testing {len(prompt_names)} prompts × {args.repeats} repeats = {len(prompt_names)*args.repeats} runs")

    # 运行测试
    results = run_test(topic_text, prompt_names, args.repeats)

    # 输出汇总
    print_summary(results)

    # 保存详细结果
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
