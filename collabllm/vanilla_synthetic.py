import json
import logging
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from collabllm.simulation import ChatSessionSimulator
from collabllm.singleturn_reward import singleturn_reward
from collabllm.utils.template import strip_system_prompt

logger = logging.getLogger(__name__)


def generate_singleturn_dataset(
    *,
    task_desc: str,
    single_turn_prompt: str,
    single_turn_completion: str,
    single_turn_metadata: Dict[str, Any],
    metric_names: List[str],
    user_generation_kwargs: Dict[str, Any],
    assistant_generation_kwargs: Dict[str, Any],
    reward_generation_kwargs: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[List[float]] = None,
    proact_prompt_ratio: float = 0.5,
    add_system_prompt_ratio: float = 0.5,
    local_model: Optional[AutoModelForCausalLM] = None,
    local_tokenizer: Optional[AutoTokenizer] = None,
    vllm_base_model: Optional[Any] = None,
    num_candidate_responses: int = 5,
    max_total_turns: int = 8,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    为 Vanilla DPO 实验生成嵌套格式的合成对话数据。
    与 `generate_multiturn_dataset` 的区别：
    - 不会循环模拟未来的多轮对话。
    - 使用 `singleturn_reward` 仅对当前候选回复进行评分。
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)

    sim = ChatSessionSimulator()
    chat_history: List[Dict[str, str]] = []

    # 共享的模拟参数
    base_sim_args = {
        "task_desc": task_desc,
        "single_turn_prompt": single_turn_prompt,
        "local_model": local_model,
        "local_tokenizer": local_tokenizer,
        "vllm_base_model": vllm_base_model,
        "assistant_generation_kwargs": assistant_generation_kwargs,
        "user_generation_kwargs": user_generation_kwargs,
    }

    # 返回的嵌套结构
    multiturn_data: Dict[str, Any] = {
        "single_turn_prompt": single_turn_prompt,
        "single_turn_completion": single_turn_completion,
        "single_turn_metadata": single_turn_metadata,
        "turns": [],
    }

    # 1) 初始用户轮
    first_user_msg = sim.run_chat_simulation(
        **base_sim_args,
        num_samples=1,
        chat_history=chat_history,
        max_new_turns=1,
        max_workers=1,
        verbose=False,
    )[0][-1]
    chat_history.append(first_user_msg)

    # 2) 循环生成直到达到最大轮数
    while len(chat_history) < max_total_turns:
        # a) 采样助手候选回复
        candidate_hists = sim.run_chat_simulation(
            **base_sim_args,
            proact_prompt_ratio=proact_prompt_ratio,
            num_samples=num_candidate_responses,
            chat_history=chat_history,
            add_system_prompt_ratio=add_system_prompt_ratio,
            max_workers=max_workers,
            max_new_turns=1,  # 仅生成下一条助手回复
            verbose=False,
            log_prefix="[Candidate] ",
        )

        candidate_completions = [
            hist[-1]["content"] for hist in candidate_hists if hist
        ]

        if not candidate_completions:
            logger.warning("No candidate completions generated. Terminating early.")
            break

        # b) 使用单轮奖励对每个候选进行评分（不模拟未来）
        turn_prompt = list(chat_history)  # 复制到用户轮为止的历史
        responses_with_scores: List[Dict[str, Any]] = []
        scores: List[float] = []

        for completion in candidate_completions:
            temp_history = chat_history + [{"role": "assistant", "content": completion}]
            rewards = singleturn_reward(
                single_turn_prompt=single_turn_prompt,
                single_turn_completion=single_turn_completion,
                metric_names=metric_names,
                chat_history=temp_history,
                reward_generation_kwargs=reward_generation_kwargs,
                metadata=single_turn_metadata,
                metric_weights=metric_weights,
            )
            score = rewards.get("MR", 0.0)

            responses_with_scores.append({
                "completion": completion,
                "score": score,
                "rewards": rewards,
            })
            scores.append(score)

        logger.info(
            f"\n\nResponses and single-turn scores (Turn {len(chat_history) // 2}):"
        )
        logger.info(
            json.dumps(
                [
                    {
                        "completion": r["completion"],
                        "rewards": r["rewards"],
                        "score": r["score"],
                    }
                    for r in responses_with_scores
                ],
                indent=2,
                ensure_ascii=False,
            )
        )

        multiturn_data["turns"].append({
            "prompt": strip_system_prompt(turn_prompt.copy()),
            "responses": responses_with_scores,
        })

        # c) 选取得分最高的助手回复，继续对话线程
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_response = responses_with_scores[best_idx]["completion"]
        chat_history.append({"role": "assistant", "content": best_response})

        if len(chat_history) >= max_total_turns:
            break

        # d) 基于最佳助手回复模拟下一条用户回复
        next_user_hists = sim.run_chat_simulation(
            **base_sim_args,
            num_samples=1,
            chat_history=chat_history,
            max_new_turns=1,
            max_workers=1,
            verbose=False,
        )

        if not next_user_hists or not next_user_hists[0]:
            logger.warning("未能生成下一条用户回复，提前终止。")
            break

        next_user_msg = next_user_hists[0][-1]
        chat_history.append({"role": "user", "content": next_user_msg["content"]})

        if sim._should_terminate_conversation(next_user_msg["content"]):
            logger.info("Conversation terminated by user.")
            break

    return multiturn_data
