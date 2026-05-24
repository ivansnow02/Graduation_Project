"""
多轮奖励计算（一次调用 ChatSessionSimulator 即可返回多条会话）。

假设：
• ChatSessionSimulator.run_chat_simulation 支持 `num_samples`，一次返回多条会话（内部并行/批处理）。
• SingleTurnOrChatMetric 接口保持不变。
"""

from __future__ import annotations

import logging
import statistics as stats
from typing import Any, Dict, List, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from collabllm.metric import SingleTurnOrChatMetric
from collabllm.simulation import ChatSessionSimulator
from collabllm.utils.template import strip_system_prompt


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric 辅助函数                                                               #
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
# 辅助：美观的统计汇总表                                                         #
# --------------------------------------------------------------------------- #
def _log_reward_summary(reward_dict: Dict[str, List[float]]) -> None:
    """Compute mean / std for each metric list in `reward_dict` and log."""
    rows = []
    for metric, vals in reward_dict.items():
        # vals 在评估后总是列表形式（包括 "MR"）
        mu = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((metric, f"{mu:.3f}", f"{sd:.3f}"))

    header = ("Metric", "Mean", "Std")

    try:
        from tabulate import tabulate

        table = "\n" + tabulate(rows, headers=header, tablefmt="github")
    except ImportError:
        colw = [max(len(x) for x in col) for col in zip(*([header] + rows))]
        fmt = "  ".join(f"{{:<{w}}}" for w in colw)
        table = (
            "\n" + fmt.format(*header) + "\n" + "\n".join(fmt.format(*r) for r in rows)
        )

    logger.info("Reward statistics:%s", table)


# --------------------------------------------------------------------------- #
# 公共 API                                                                     #
# --------------------------------------------------------------------------- #
def multiturn_aware_reward(
    *,
    task_desc: str,
    single_turn_prompt: str,
    single_turn_completion: str,
    metric_names: Sequence[str],
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
    max_metric_workers: int = 16,
    return_details: bool = False,
    **chat_simulation_kwargs,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Any]]]:
    """
    计算一次批量返回的多条会话的奖励。
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")

    # ------------------------------------------------------------------ #
    # 1 · 一次性生成所有会话                                                 #
    # ------------------------------------------------------------------ #
    sessions = ChatSessionSimulator().run_chat_simulation(
        task_desc=task_desc,
        single_turn_prompt=single_turn_prompt,
        log_prefix="[Deduction] ",
        **chat_simulation_kwargs,
    )  # → List[List[dict]]
    # 去除可能存在的 system 消息
    sessions = [strip_system_prompt(session) for session in sessions]

    # ------------------------------------------------------------------ #
    # 2 · 准备结果容器                                                      #
    # ------------------------------------------------------------------ #
    reward_dict: Dict[str, List[float]] = {m: [] for m in metric_names}
    reward_dict["MR"] = []

    # ------------------------------------------------------------------ #
    # 3 · 指标评估（对每个会话 × 指标进行完全并行计算）                      #
    # ------------------------------------------------------------------ #
    n_conv = len(sessions)
    # 初始化存储结构
    for m in metric_names:
        reward_dict[m] = [0.0] * n_conv
    reward_dict["MR"] = [0.0] * n_conv

    with ThreadPoolExecutor(max_workers=max_metric_workers) as pool:
        fut_to_ctx = {}
        for conv_idx, messages in enumerate(sessions):
            for i, metric_name in enumerate(metric_names):
                fut = pool.submit(
                    _score_one_metric,
                    metric_name,
                    messages,
                    reward_generation_kwargs,
                    single_turn_prompt,
                    single_turn_completion,
                    metadata,
                )
                # 保存上下文：对应哪个会话 / 哪个指标 / 权重索引
                fut_to_ctx[fut] = (conv_idx, i, metric_name)

        for fut in as_completed(fut_to_ctx):
            conv_idx, i, metric_name = fut_to_ctx[fut]
            score = fut.result()
            reward_dict[metric_name][conv_idx] = score

    # ------------------------------------------------------------------ #
    # 4 · 聚合 → 计算 Multiturn-aware Reward (MR)
    # ------------------------------------------------------------------ #
    for conv_idx in range(n_conv):
        reward_dict["MR"][conv_idx] = sum(
            reward_dict[m][conv_idx] * metric_weights[i]
            for i, m in enumerate(metric_names)
        )
    _log_reward_summary(reward_dict)
    if return_details:
        return reward_dict, sessions
    return reward_dict
