# 正则规范化函数：将各种变体映射到标准类型


import re
from typing import Optional


def normalize_cognitive_level(text: str) -> int:
    """
    将各种认知层级文本规范化为 Bloom 等级 (1-6)。
    基于 analyze_fields.py 的输出，覆盖中文、英文、括号变体。

    优先级：精确匹配 > 模式匹配 > 默认返回 0
    """
    if not text:
        return 0

    text = text.strip().lower()

    # 创造 / 生成 (Bloom 6)
    if re.search(r"创造|生成|create|generate", text):
        return 6

    # 评价 / 评估 (Bloom 5)
    if re.search(r"评价|评估|evaluate", text):
        return 5

    # 分析 (Bloom 4)
    if re.search(r"分析|分解|analyze|synthesis", text):
        return 4

    # 应用 (Bloom 3)
    if re.search(r"应用|apply|practice", text):
        return 3

    # 理解 (Bloom 2)
    if re.search(r"理解|understand|explain|interpret", text):
        return 2

    # 记忆 (Bloom 1)
    if re.search(r"记忆|记住|remember|recall|identify", text):
        return 1

    return 0


def canonicalize_teaching_intent(text: str) -> Optional[str]:
    """
    将教学意图文本规范化为 4 大标准类别之一（或 None 如果空白/未知）。
    四大类别：introduce, check_understanding, guide_reasoning, summarize_enhance
    """
    if not text or not text.strip():
        return None

    text = text.strip().lower()

    # 1. 引入概念 (Introduce)
    if re.search(r"引入|引出|介绍|introduce|present|initiate|开始", text):
        return "introduce"

    # 2. 检测理解 (Check Understanding)
    if re.search(r"检测|检查|理解|核实|understand|verify|check|assess", text):
        return "check_understanding"

    # 3. 引导推理 (Guide Reasoning)
    if re.search(r"引导|推理|推导|思考|guide|reason|deduce|infer|analyze", text):
        return "guide_reasoning"

    # 4. 总结提升 (Summarize and Enhance)
    if re.search(r"总结|汇总|提升|归纳|consolidate|summarize|enhance|conclude", text):
        return "summarize_enhance"

    return None


def canonicalize_teaching_strategy(text: str) -> Optional[str]:
    """
    将教学策略文本规范化为 9 大类别之一（或 None 如果空白/未知）。
    类别（按优先级排序）：
      0. invalid_echo - 无效复读（最高优先级，用于惩罚复读机）
      1. scenario_questioning - 情境设问、场景问题
      2. follow_up_questioning - 追问、延伸提问
      3. analogies - 类比、类比法
      4. hints - 提示、暗示、启发
      5. break_down - 拆解、分解问题
      6. encourage_responses - 鼓励、正向反馈
      7. corrective_feedback - 纠正、错误反馈
      8. reasoning_guidance - 引导推理、迁移、导向
    """
    if not text or not text.strip():
        return None

    text = text.strip().lower()

    # 0. 无效复读 (Invalid Echo) — 最高优先级，防止被其他规则吞没
    if re.search(r"无效复读|复读|echo|parrot|repeat", text):
        return "invalid_echo"

    # 1. 情境设问 (Scenario-based questioning)
    if re.search(r"情境|场景|scenario|context|situate", text):
        return "scenario_questioning"

    # 2. 追问 (Follow-up questioning) - 优先级高，防止被后续规则吞没
    if re.search(r"追问|续问|follow.?up|extend|further|probe", text):
        return "follow_up_questioning"

    # 3. 类比 (Analogies)
    if re.search(r"类比|比喻|analogies?|metaphor|simile", text):
        return "analogies"

    # 4. 提示 (Hints)
    if re.search(r"提示|暗示|启发|hints?|cue|clue|suggestion", text):
        return "hints"

    # 5. 拆解 (Break down complex questions)
    if re.search(r"拆解|分解|拆分|break.?down|decompose|simplify", text):
        return "break_down"

    # 6. 鼓励回应 (Encourage responses)
    if re.search(r"鼓励|激励|正向|encourage|praise|affirm|respond", text):
        return "encourage_responses"

    # 7. 纠正反馈 (Corrective feedback)
    if re.search(r"纠正|正误|反馈|纠|corrective?|feedback|error|wrong|incorrect", text):
        return "corrective_feedback"

    # 8. 引导推理/引发迁移/思考导向 (Reasoning guidance)
    if re.search(r"引导|引发|迁移|思考|导向|guide|reason|transfer|think|infer", text):
        return "reasoning_guidance"

    return None
