"""
collabllm.utils.metrics
~~~~~~~~~~~~~~~~~~~~~~~
教学会话评分的共享评估指标集合。
与 `scripts/benchmark/objective_eval.py` 保持对齐。
"""

import re
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

from collabllm.datasets.types import TeachingSession, Annotation
from collabllm.utils.clean import (
    canonicalize_teaching_intent,
    canonicalize_teaching_strategy,
    normalize_cognitive_level,
)


# 来自 objective_eval.py 的常量
TOTAL_POSSIBLE_STRATEGIES = 8
MAX_BLOOM_PROGRESSION = 5  # 6 - 1

# 标准权重定义
WEIGHTS = {
    "strategy_density": 0.15,
    "strategy_variety": 0.10,
    "ikt": 0.15,  # 跨学科知识迁移（Interdisciplinary Knowledge Transfer）
    "bloom_progression": 0.15,
    "structure_completeness": 0.15,
    "l3_guidance_rate": 0.10,
    "cognitive_correction_rate": 0.20,
}


def calculate_session_metrics(session: TeachingSession) -> Dict[str, Any]:
    """
    计算一次教学会话的综合质量指标。

    Returns:
        包含各项指标分数以及最终总分的字典。

    其中包括：
        - 各单项指标分数（原始值与归一化值）
        - total_score：最终加权质量分数（0-1）
    """
    annotations = session.annotations
    if not annotations:
        return {"total_score": 0.0}

    # 辅助过滤器
    teacher_anns = [a for a in annotations if a.is_teacher_annotation()]
    student_anns = [a for a in annotations if a.is_student_annotation()]

    # 需要统计教师发言的数量（注意：一次发言可能对应多个标注）。
    # 在 objective_eval.py 中，通常通过从标注列表中过滤出教师发言来计算。
    # 这里我们直接使用教师标注的数量作为近似值。
    teacher_ann_count = len(teacher_anns)

    # 1. 策略密度与多样性
    # 密度：包含至少一种策略的教师回合占比。
    # 多样性：涵盖的标准策略数量占比（最多 8 种）。

    turns_with_strategy = 0
    unique_strategies_found = set()

    for ann in teacher_anns:
        raw_strat = ann.teaching_strategy
        if not raw_strat:
            continue

        # 如果用逗号分隔，解析多个策略
        current_turn_has_strategy = False
        parts = raw_strat.replace("，", ",").split(",")
        for part in parts:
            p = part.strip()
            if not p:
                continue
            canonical = canonicalize_teaching_strategy(p)
            if canonical:
                unique_strategies_found.add(canonical)
                current_turn_has_strategy = True

        if current_turn_has_strategy:
            turns_with_strategy += 1

    strategy_density = (
        (turns_with_strategy / teacher_ann_count) if teacher_ann_count > 0 else 0.0
    )
    strategy_variety = len(unique_strategies_found) / TOTAL_POSSIBLE_STRATEGIES
    strategy_variety = min(strategy_variety, 1.0)

    # 2. 跨学科知识迁移（IKT）
    # objective_eval：统计 discipline_transfer 字段为 "是" 的回合数
    # 得分 = count / 教师回合总数

    transfer_count = 0
    for ann in teacher_anns:
        # 检查标准正值
        val = str(ann.discipline_transfer).strip().lower()
        if val in ["是", "yes", "true", "1"]:
            transfer_count += 1

    ikt_score = (transfer_count / teacher_ann_count) if teacher_ann_count > 0 else 0.0
    ikt_score = min(ikt_score, 1.0)  # 上限限定为 1.0

    # 3. 结构完整性（Structure Completeness）
    # objective_eval 使用的四个桶：{"引出概念", "引导推理", "引发迁移", "总结提升"}
    # 我们将标准化后的意图映射到这些类别，并另外检测迁移（Transfer）。

    detected_structure_buckets = set()

    for ann in teacher_anns:
        raw_intent = ann.teacher_intent
        if not raw_intent:
            continue

        # 使用已有的意图规范化器进行映射
        canonical = canonicalize_teaching_intent(raw_intent)

        if canonical == "introduce":
            detected_structure_buckets.add("引出概念")
        elif canonical == "guide_reasoning":
            detected_structure_buckets.add("引导推理")
        elif canonical == "summarize_enhance":
            detected_structure_buckets.add("总结提升")
        elif canonical == "check_understanding":
            # 说明：objective_eval 在其 4 项集合中不严格要求 Check Understanding。
            # 但它是有效意图；如果严格按 objective_eval 的要求集合计算，则忽略它。
            pass

        # 特殊检测：如果原始意图文本中包含迁移相关词，则标记为 "引发迁移"
        # 注意：canonicalize_teaching_intent 并未把迁移显式作为一种类型返回。
        if "迁移" in raw_intent or "transfer" in raw_intent.lower():
            detected_structure_buckets.add("引发迁移")

    # objective_eval 要求覆盖的意图集合
    REQUIRED_INTENTS = {"引出概念", "引导推理", "引发迁移", "总结提升"}
    covered_count = len(detected_structure_buckets.intersection(REQUIRED_INTENTS))
    structure_completeness = covered_count / len(REQUIRED_INTENTS)

    # 4. L3 指导率
    # objective_eval：teacher_guidance_level == "L3"

    l3_count = 0
    for ann in teacher_anns:
        if "L3" in str(ann.teacher_guidance_level):
            l3_count += 1

    l3_guidance_rate = (l3_count / teacher_ann_count) if teacher_ann_count > 0 else 0.0

    # 5. Bloom 进阶（Bloom Progression）
    # objective_eval：计算 (max - min) / (6-1)

    student_bloom_levels = []
    for ann in student_anns:
        # 使用 utils.clean.normalize_cognitive_level 将文本转换为 Bloom 等级
        # 注意：clean.py 的函数接收文本，返回整数（0-6）
        # ann.cognitive_level 通常为字符串文本
        level = normalize_cognitive_level(ann.cognitive_level)
        if level > 0:  # 有效等级
            student_bloom_levels.append(level)

    if student_bloom_levels:
        raw_bp = max(student_bloom_levels) - min(student_bloom_levels)
        bloom_progression = raw_bp / MAX_BLOOM_PROGRESSION
    else:
        bloom_progression = 0.0
    bloom_progression = min(bloom_progression, 1.0)

    # 6. 认知纠正率（Cognitive Correction Rate, 3C）
    # objective_eval：successful_correction_count / total_error_count
    # 错误判定关键词示例："错误回答"
    # 修正成功的判定：从错误状态 -> 下一状态为 ["高阶思考", "清晰理解"]
    # 需要按时间顺序遍历学生状态序列。
    # 通常 TeachingSession.annotations 会按出现顺序排列。

    student_states = []
    for ann in student_anns:
        student_states.append(str(ann.student_cognition_state))

    total_error_count = 0
    successful_correction_count = 0

    # 定义基于 clean.py / objective_eval 的关键词集合
    # objective_eval 使用精确词条："错误回答", "高阶思考", "清晰理解"
    # 实际数据可能存在变体，因此使用更鲁棒的关键词匹配。

    def is_error(s):
        return any(k in s for k in ["错误", "Incorrect", "Error", "Vague", "模糊"])

    def is_clear(s):
        return any(k in s for k in ["清晰", "Clear", "高阶", "Higher-order"])

    # 第一遍：统计错误
    total_error_count = sum(1 for s in student_states if is_error(s))

    # 第二遍：统计修正
    for i in range(len(student_states) - 1):
        current_s = student_states[i]
        next_s = student_states[i + 1]

        if is_error(current_s) and is_clear(next_s):
            successful_correction_count += 1

    if total_error_count == 0:
        cognitive_correction_rate = 1.0  # 若无错误则默认满分
    else:
        cognitive_correction_rate = successful_correction_count / total_error_count

    cognitive_correction_rate = min(cognitive_correction_rate, 1.0)

    # 计算总分
    total_score = (
        WEIGHTS["strategy_density"] * strategy_density
        + WEIGHTS["strategy_variety"] * strategy_variety
        + WEIGHTS["ikt"] * ikt_score
        + WEIGHTS["bloom_progression"] * bloom_progression
        + WEIGHTS["structure_completeness"] * structure_completeness
        + WEIGHTS["l3_guidance_rate"] * l3_guidance_rate
        + WEIGHTS["cognitive_correction_rate"] * cognitive_correction_rate
    )

    return {
        "strategy_density": strategy_density,
        "strategy_variety": strategy_variety,
        "ikt_score": ikt_score,
        "bloom_progression": bloom_progression,
        "structure_completeness": structure_completeness,
        "l3_guidance_rate": l3_guidance_rate,
        "cognitive_correction_rate": cognitive_correction_rate,
        "total_score": round(total_score, 4),
    }


def calculate_turn_metrics(
    turn: Annotation, history_annotations: List[Annotation]
) -> Dict[str, float]:
    """
    基于单个教师回合的标注与历史上下文计算奖励。
    返回每个维度的二值/类别分数（0.0 或 1.0）。

    增强项：检测 "无效复读" 策略并施加惩罚。
    """
    # 0. 无效复读检测 — 如果当前回合标记为 "invalid_echo"，则视为复读并惩罚
    is_invalid_echo = False
    if turn.teaching_strategy:
        raw_parts = turn.teaching_strategy.replace("\uff0c", ",").split(",")
        for part in raw_parts:
            p = part.strip()
            if p:
                canonical = canonicalize_teaching_strategy(p)
                if canonical == "invalid_echo":
                    is_invalid_echo = True
                    break

    # 1. 策略密度（是否存在策略）
    # 无效复读不计为有效策略
    has_strategy = 0.0 if is_invalid_echo else (1.0 if turn.teaching_strategy else 0.0)

    # 2. 策略多样性（是否为新策略）
    observed_strategies = set()
    for ann in history_annotations:
        if ann.is_teacher_annotation() and ann.teaching_strategy:
            parts = ann.teaching_strategy.replace("\uff0c", ",").split(",")
            for part in parts:
                p = part.strip()
                if p:
                    canonical = canonicalize_teaching_strategy(p)
                    if canonical:
                        observed_strategies.add(canonical)

    current_strategies = set()
    if turn.teaching_strategy:
        parts = turn.teaching_strategy.replace("\uff0c", ",").split(",")
        for part in parts:
            p = part.strip()
            if p:
                canonical = canonicalize_teaching_strategy(p)
                if canonical:
                    current_strategies.add(canonical)

    is_new_strategy = 1.0 if (current_strategies - observed_strategies) else 0.0

    # 3. IKT（是否存在迁移）
    has_transfer = (
        1.0
        if str(turn.discipline_transfer).strip().lower() in ["是", "yes", "true", "1"]
        else 0.0
    )

    # 4. 结构完整性（是否为新的意图类型）
    observed_intents = set()
    for ann in history_annotations:
        if ann.is_teacher_annotation() and ann.teacher_intent:
            canonical = canonicalize_teaching_intent(ann.teacher_intent)
            if canonical:
                observed_intents.add(canonical)

    current_intent = (
        canonicalize_teaching_intent(turn.teacher_intent)
        if turn.teacher_intent
        else None
    )
    is_new_intent = (
        1.0 if (current_intent and current_intent not in observed_intents) else 0.0
    )

    # 5. L3 指导（是否为 L3）
    # 如果为无效复读，即使被标注为 L3 也强制视为 0
    is_l3 = (
        0.0
        if is_invalid_echo
        else (1.0 if "L3" in str(turn.teacher_guidance_level) else 0.0)
    )

    # 加权求和（权重说明）
    # Density: 0.25, Variety: 0.15, IKT: 0.2, Structure: 0.2, L3: 0.2
    weights = {
        "strategy_density": 0.25,
        "strategy_variety": 0.15,
        "ikt": 0.2,
        "structure_completeness": 0.2,
        "l3_guidance_rate": 0.2,
    }

    total_score = (
        weights["strategy_density"] * has_strategy
        + weights["strategy_variety"] * is_new_strategy
        + weights["ikt"] * has_transfer
        + weights["structure_completeness"] * is_new_intent
        + weights["l3_guidance_rate"] * is_l3
    )

    # 无效复读额外惩罚：即使其他维度有分，也要压低总分
    if is_invalid_echo:
        total_score = max(total_score - 0.15, 0.0)

    return {
        "strategy_density": has_strategy,
        "strategy_variety": is_new_strategy,
        "ikt_score": has_transfer,
        "structure_completeness": is_new_intent,
        "l3_guidance_rate": is_l3,
        "is_invalid_echo": 1.0 if is_invalid_echo else 0.0,
        "total_score": round(total_score, 4),
    }
