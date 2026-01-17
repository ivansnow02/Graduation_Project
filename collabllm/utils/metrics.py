"""
collabllm.utils.metrics
~~~~~~~~~~~~~~~~~~~~~~~
Shared evaluation metrics for scoring teaching sessions.
Aligned with `scripts/benchmark/objective_eval.py`.
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


# Constants from objective_eval.py
TOTAL_POSSIBLE_STRATEGIES = 8
MAX_BLOOM_PROGRESSION = 5  # 6 - 1

# Standard weights
WEIGHTS = {
    "strategy_density": 0.15,
    "strategy_variety": 0.10,
    "ikt": 0.15,  # Interdisciplinary Knowledge Transfer
    "bloom_progression": 0.15,
    "structure_completeness": 0.15,
    "l3_guidance_rate": 0.10,
    "cognitive_correction_rate": 0.20,
}


def calculate_session_metrics(session: TeachingSession) -> Dict[str, Any]:
    """
    Calculate comprehensive quality metrics for a teaching session.

    Returns a dictionary containing:
      - Individual metric scores (raw and normalized)
      - 'total_score': The final weighted quality score (0-1)
    """
    annotations = session.annotations
    if not annotations:
        return {"total_score": 0.0}

    # Helper filters
    teacher_anns = [a for a in annotations if a.is_teacher_annotation()]
    student_anns = [a for a in annotations if a.is_student_annotation()]

    # We need teacher utterances count. Note: One utterance might have multiple annotations?
    # objective_eval.py: teacher_utterance_count = len(teacher_utterances) where teacher_utterances is filtered from annotations list.
    # We usually trust the annotation list length for this calculation in objective_eval.
    teacher_ann_count = len(teacher_anns)

    # --- 1. Strategy Density & Variety ---
    # Density: Proportion of teacher turns containing at least one strategy.
    # Variety: Coverage of the 8 standard strategies.

    turns_with_strategy = 0
    unique_strategies_found = set()

    for ann in teacher_anns:
        raw_strat = ann.teaching_strategy
        if not raw_strat:
            continue

        # Parse multiple strategies if comma-separated
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

    # --- 2. Interdisciplinary Knowledge Transfer (IKT) ---
    # objective_eval: count where discipline_transfer == "是"
    # Score = count / total_teacher_turns

    transfer_count = 0
    for ann in teacher_anns:
        # Check standard positive values
        val = str(ann.discipline_transfer).strip().lower()
        if val in ["是", "yes", "true", "1"]:
            transfer_count += 1

    ikt_score = (transfer_count / teacher_ann_count) if teacher_ann_count > 0 else 0.0
    ikt_score = min(ikt_score, 1.0)  # Cap at 1.0 just in case

    # --- 3. Structure Completeness ---
    # objective_eval uses: {"引出概念", "引导推理", "引发迁移", "总结提升"}
    # We map canonical English intents to these buckets plus check for 'Transfer'.

    detected_structure_buckets = set()

    for ann in teacher_anns:
        raw_intent = ann.teacher_intent
        if not raw_intent:
            continue

        # Use existing canonicalizer
        canonical = canonicalize_teaching_intent(raw_intent)

        if canonical == "introduce":
            detected_structure_buckets.add("引出概念")
        elif canonical == "guide_reasoning":
            detected_structure_buckets.add("引导推理")
        elif canonical == "summarize_enhance":
            detected_structure_buckets.add("总结提升")
        elif canonical == "check_understanding":
            # Note: objective_eval doesn't strictly require Check Understanding in its set of 4.
            # But it's a valid intent. We ignore it for the *Required Set* calculation if strictly following objective_eval.
            pass

        # Special check for "引发迁移" (Transfer) logic in Intent
        # objective_eval looks for it in intent field.
        # canonicalize_teaching_intent doesn't capture it explicitly as a type.
        if "迁移" in raw_intent or "transfer" in raw_intent.lower():
            detected_structure_buckets.add("引发迁移")

    # objective_eval required set
    REQUIRED_INTENTS = {"引出概念", "引导推理", "引发迁移", "总结提升"}
    covered_count = len(detected_structure_buckets.intersection(REQUIRED_INTENTS))
    structure_completeness = covered_count / len(REQUIRED_INTENTS)

    # --- 4. L3 Guidance Rate ---
    # objective_eval: teacher_guidance_level == "L3"

    l3_count = 0
    for ann in teacher_anns:
        if "L3" in str(ann.teacher_guidance_level):
            l3_count += 1

    l3_guidance_rate = (l3_count / teacher_ann_count) if teacher_ann_count > 0 else 0.0

    # --- 5. Bloom Progression (BP) ---
    # objective_eval: (max - min) / (6-1)

    student_bloom_levels = []
    for ann in student_anns:
        # Use utils.clean.normalize_cognitive_level
        # Note: clean.py function takes 'text', returns int (0-6)
        # ann.cognitive_level is the string text
        level = normalize_cognitive_level(ann.cognitive_level)
        if level > 0:  # valid level
            student_bloom_levels.append(level)

    if student_bloom_levels:
        raw_bp = max(student_bloom_levels) - min(student_bloom_levels)
        bloom_progression = raw_bp / MAX_BLOOM_PROGRESSION
    else:
        bloom_progression = 0.0
    bloom_progression = min(bloom_progression, 1.0)

    # --- 6. Cognitive Correction Rate (3C) ---
    # objective_eval: successful_correction_count / total_error_count
    # Error: "错误回答"
    # Success: Error -> (Next State in ["高阶思考", "清晰理解"])

    # We need to traverse states in order.
    # student_anns should be in chronological order ideally.
    # TeachingSession.annotations is a list, usually ordered?
    # Yes, typically ordered by appearance.

    student_states = []
    for ann in student_anns:
        student_states.append(str(ann.student_cognition_state))

    total_error_count = 0
    successful_correction_count = 0

    # Define keywords based on clean.py/objective_eval
    # objective_eval uses exact strings: "错误回答", "高阶思考", "清晰理解"
    # But clean data might have variations.
    # We'll use robust keywords.

    def is_error(s):
        return any(k in s for k in ["错误", "Incorrect", "Error", "Vague", "模糊"])

    def is_clear(s):
        return any(k in s for k in ["清晰", "Clear", "高阶", "Higher-order"])

    # First pass: count errors
    total_error_count = sum(1 for s in student_states if is_error(s))

    # Second pass: count corrections
    for i in range(len(student_states) - 1):
        current_s = student_states[i]
        next_s = student_states[i + 1]

        if is_error(current_s) and is_clear(next_s):
            successful_correction_count += 1

    if total_error_count == 0:
        cognitive_correction_rate = 1.0  # Default full score if no errors made
    else:
        cognitive_correction_rate = successful_correction_count / total_error_count

    cognitive_correction_rate = min(cognitive_correction_rate, 1.0)

    # --- Total Score ---
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
