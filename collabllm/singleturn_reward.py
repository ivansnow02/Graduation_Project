"""
collabllm.singleturn_reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~
单轮奖励计算工具。

该模块用于只评估当前回复质量的场景，常见于单轮训练与 DPO 消融实验。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from collabllm.metric import SingleTurnOrChatMetric
from collabllm.utils.template import strip_system_prompt

logger = logging.getLogger(__name__)


# Metric helper
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


# 公共 API
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
    计算以候选回复结尾的单条对话历史的奖励。
    返回一个字典，包含每个 metric 的分数及合成的 "MR" 分数。
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")

    # 1. 去除 system 消息
    messages = strip_system_prompt(chat_history)

    # 2. 计算各项指标得分
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
