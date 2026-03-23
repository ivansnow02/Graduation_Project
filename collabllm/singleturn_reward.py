"""
Single-turn reward computation for DPO ablation experiments.

Assumes:
• We do not simulate future turns to get a "multiturn-aware" score.
• We evaluate the quality of the immediate response given the chat history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from collabllm.metric import SingleTurnOrChatMetric
from collabllm.utils.template import strip_system_prompt

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric helper                                                               #
# --------------------------------------------------------------------------- #
def _score_one_metric(
    metric_name: str,
    messages: List[Dict[str, str]],
    metric_kwargs: Dict[str, Any],
    prompt: str,
    completion: str,
    metadata: Dict[str, Any] | None,
) -> float:
    metric = SingleTurnOrChatMetric(signature=metric_name, **metric_kwargs)
    res = metric(
        messages=messages,
        single_turn_prompt=prompt,
        single_turn_completion=completion,
        metadata=metadata,
    )
    from typing import cast

    return cast(float, res)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def singleturn_reward(
    *,
    single_turn_prompt: str,
    single_turn_completion: str,
    metric_names: Sequence[str],
    chat_history: List[Dict[str, str]],  # 已包含了候选的 assistant response
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
) -> Dict[str, float]:
    """
    Compute rewards for a single conversation history ending with a candidate response.
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")

    # 1. Strip system message
    messages = strip_system_prompt(chat_history)

    # 2. Evaluate metrics
    reward_dict: Dict[str, float] = {}
    mr_score = 0.0

    for i, metric_name in enumerate(metric_names):
        score = _score_one_metric(
            metric_name,
            messages,
            reward_generation_kwargs,
            single_turn_prompt,
            single_turn_completion,
            metadata,
        )
        reward_dict[metric_name] = score
        mr_score += score * metric_weights[i]

    reward_dict["MR"] = mr_score
    
    return reward_dict
