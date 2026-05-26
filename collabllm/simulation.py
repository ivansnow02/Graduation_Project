"""聊天会话模拟器。"""
from __future__ import annotations

import os
import copy
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import random
import torch
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from collabllm.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SOCRATIC_STRICT,
    SYSTEM_PROMPT_DIRECT_ANSWER,
    SYSTEM_PROMPT_IMPATIENT_LEAKY,
)
from collabllm import ENABLE_COLLABLLM_LOGGING
from collabllm.prompts import COLLABLLM_TERMINATION_SIGNAL
from collabllm.modules import LLMCollaborator, UserSimulator
from collabllm.utils.format import is_conversational

logger = logging.getLogger(__name__)

PEFT_CHECKPOINT_SUFFIX = "-peft-checkpoint"

# 有效的 vLLM 采样参数
VALID_VLLM_SAMPLING_PARAMS = {
    "n",
    "best_of",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "stop_token_ids",
    "bad_words",
    "ignore_eos",
    "max_tokens",
    "min_tokens",
    "logprobs",
    "prompt_logprobs",
    "detokenize",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "truncate_prompt_tokens",
}


class ChatSessionSimulator:
    """管理多个并发的聊天会话。"""

    def run_chat_simulation(
        self,
        *,
        task_desc: str,
        single_turn_prompt: str,
        chat_history: List[Dict[str, str]],
        assistant_generation_kwargs: Dict[str, Any],
        user_generation_kwargs: Dict[str, Any],
        num_samples: int = 1,
        max_new_turns: int = 0,
        proact_prompt_ratio: int = 0.0,
        add_system_prompt_ratio: float = 0.0,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        vllm_base_model: Optional = None,
        max_workers: int = 8,
        verbose: bool = True,
        log_prefix: str = "",
    ) -> List[List[Dict[str, str]]]:
        """
        并行模拟 `num_samples` 条对话（内部使用批处理/并发）。

        Returns:
            长度为 `num_samples` 的完整对话转录列表。
        """
        # 0. 参数校验与默认值
        self._validate_session_inputs(
            task_desc,
            single_turn_prompt,
            max_new_turns,
            local_model,
            local_tokenizer,
            vllm_base_model,
            assistant_generation_kwargs,
            user_generation_kwargs,
        )

        # 1. 每个会话的初始状态
        sessions: List[List[Dict[str, str]]] = [
            copy.deepcopy(chat_history or []) for _ in range(num_samples)
        ]
        # 为了增加样本的多样性，在部分会话中注入 system prompt
        for sess in sessions[: int(num_samples * add_system_prompt_ratio)]:
            sess.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # 验证 system prompt 是否已正确插入
        if sessions and len(sessions) > 0:
            has_system = len(sessions[0]) > 0 and sessions[0][0].get("role") == "system"
            logger.info(
                f"DEBUG: run_chat_simulation Start. Num Samples: {num_samples}, Add Ratio: {add_system_prompt_ratio}, System Prompt Present @0? {has_system}"
            )

        current_roles = [self._determine_starting_role(hist) for hist in sessions]
        active: set[int] = set(range(num_samples))  # 当前仍处于活动状态的会话索引

        user_sims = [
            UserSimulator(
                task_desc=task_desc,
                single_turn_prompt=single_turn_prompt,
                **user_generation_kwargs,
            )
            for _ in range(num_samples)
        ]

        # 可选：为 vLLM 准备 PEFT 检查点（若本地模型包含 peft_config）
        model_name = assistant_generation_kwargs.get("model")
        if (
            vllm_base_model is not None
            and local_model is not None
            and hasattr(local_model, "peft_config")
        ):
            self._write_peft_checkpoint(local_model, model_name)

        msg_budget = [max_new_turns for _ in range(num_samples)]  # 剩余消息预算
        active: set[int] = {i for i, b in enumerate(msg_budget) if b > 0}

        pbar = tqdm(
            total=max_new_turns,
            desc="Simulating chat",
            disable=not (ENABLE_COLLABLLM_LOGGING and verbose),
        )

        while active:
            # 用户回合
            user_idx = [i for i in active if current_roles[i] == "user"]
            if user_idx:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {
                        pool.submit(user_sims[i], sessions[i]): i for i in user_idx
                    }
                    for fut in as_completed(fut_to_i):
                        i = fut_to_i[fut]
                        resp = fut.result()
                        # 清理：删除用户模拟器返回中的状态描述
                        import re

                        resp = re.sub(r"^（\d+）.*?\n", "", resp).strip()
                        self._log_response(
                            f"user (Turn {len(sessions[i])})", resp, prefix=log_prefix
                        )
                        sessions[i].append({"role": "user", "content": resp})

                        msg_budget[i] -= 1

                        # 提前退出检查
                        if msg_budget[i] == 0 or self._should_terminate_conversation(
                            resp
                        ):
                            current_roles[i] = "terminated"
                            active.discard(i)
                        else:
                            current_roles[i] = "assistant"
                    pbar.update(1)

            if not active:  # 所有对话均耗尽预算或已终止
                break

            # 助手回合
            asst_idx = [i for i in active if current_roles[i] == "assistant"]
            if not asst_idx:
                continue

            # 生成助手回复（批量或并发）
            if local_model is None and vllm_base_model is None:
                num_asst = len(asst_idx)
                cutoff = int(num_asst * proact_prompt_ratio)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {}
                    for rank, i in enumerate(asst_idx):
                        method_i = "proact" if rank < cutoff else "none"
                        collab_i = LLMCollaborator(
                            method=method_i, **assistant_generation_kwargs
                        )
                        fut = pool.submit(collab_i, sessions[i])
                        fut_to_i[fut] = i

                    responses = {fut_to_i[f]: f.result() for f in fut_to_i}
            else:
                batch_sess = [sessions[i] for i in asst_idx]
                if vllm_base_model is not None:
                    batch_sess = self._inject_contrastive_system_prompts(batch_sess)
                    outs = self._batch_generate_with_vllm(
                        batch_sess,
                        vllm_base_model,
                        local_model,
                        model_name,
                        assistant_generation_kwargs,
                    )
                else:
                    outs = self._batch_generate_with_huggingface(
                        batch_sess,
                        local_model,
                        local_tokenizer,
                        assistant_generation_kwargs,
                    )
                responses = {g: r for g, r in zip(asst_idx, outs)}

            # 后处理助手回复
            for i, resp in responses.items():
                self._log_response(
                    f"assistant (Turn {len(sessions[i])})", resp, prefix=log_prefix
                )
                sessions[i].append({"role": "assistant", "content": resp})

                msg_budget[i] -= 1

                if msg_budget[i] == 0:
                    current_roles[i] = "terminated"
                    active.discard(i)
                else:
                    current_roles[i] = "user"
            pbar.update(1)

        pbar.close()
        return sessions

    def _inject_contrastive_system_prompts(
        self,
        batch_sess: List[List[Dict[str, str]]],
    ) -> List[List[Dict[str, str]]]:
        """为对比式候选生成注入多样化的 system persona（系统角色提示）。"""
        persona_prompts = (
            SYSTEM_PROMPT_SOCRATIC_STRICT,
            SYSTEM_PROMPT_DIRECT_ANSWER,
            SYSTEM_PROMPT_IMPATIENT_LEAKY,
        )

        injected_batch = []
        for rank, sess in enumerate(batch_sess):
            sess_copy = copy.deepcopy(sess)
            persona_prompt = persona_prompts[rank % len(persona_prompts)]

            if sess_copy and sess_copy[0].get("role") == "system":
                sess_copy[0] = {"role": "system", "content": persona_prompt}
            else:
                sess_copy.insert(0, {"role": "system", "content": persona_prompt})

            injected_batch.append(sess_copy)

        return injected_batch

    # 批量生成器
    def _batch_generate_with_vllm(
        self,
        batch_messages: List[List[Dict[str, str]]],
        vllm_base_model,
        local_model,
        model_name: str,
        generation_kwargs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> List[str]:
        """vLLM 的批量生成接口（返回回复文本列表）。"""
        from vllm.lora.request import LoRARequest

        sampling_params = self._convert_to_sampling_params(generation_kwargs)
        peft_dir = self._get_peft_dir(model_name)
        if hasattr(local_model, "peft_config") and os.path.exists(peft_dir):
            logger.info(f"Using PEFT checkpoint from {peft_dir}")
            lora_req = LoRARequest("interactive_adapter", 1, peft_dir)
        else:
            lora_req = None

        # vLLM 接受消息历史列表作为输入；返回 list[str]
        if is_conversational({"prompt": batch_messages[0]}):
            outs = vllm_base_model.chat(
                batch_messages,
                sampling_params=sampling_params,
                lora_request=lora_req,
                use_tqdm=False,
            )
        else:
            outs = vllm_base_model.generate(
                batch_messages,
                sampling_params=sampling_params,
                lora_request=lora_req,
                use_tqdm=False,
            )

        if return_outputs:
            return outs

        generated_texts = [out.outputs[0].text for out in outs]
        return generated_texts

    def _batch_generate_with_huggingface(
        self,
        batch_messages: List[List[Dict[str, str]]],
        local_model,
        local_tokenizer,
        generation_kwargs: Dict[str, Any],
    ) -> List[str]:
        """HuggingFace 的批量生成（一次前向计算）。"""
        torch.cuda.empty_cache()
        local_tokenizer.padding_side = "left"
        local_tokenizer.pad_token = local_tokenizer.eos_token

        generator = pipeline(
            "text-generation",
            model=local_model,
            tokenizer=local_tokenizer,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        generation_kwargs = copy.deepcopy(generation_kwargs)
        max_new = generation_kwargs.pop("max_tokens", 1024)
        generation_kwargs.pop("model", None)  # 对 HF pipeline 来说无需传入 model
        prompts = [msgs for msgs in batch_messages]  # HF pipeline 接受消息列表
        outputs = generator(
            prompts,
            max_new_tokens=max_new,
            **generation_kwargs,
        )

        # 提取每个输出中新生成的文本部分
        results = []
        for prompt_msgs, out in zip(prompts, outputs):
            if isinstance(out, list):
                out = out[0]  # HF pipeline 返回的是 dict 列表，取首项
            full_text = out["generated_text"]

            if isinstance(prompt_msgs, str):
                results.append(full_text[len(prompt_msgs) :])
            else:
                results.append(full_text[-1]["content"])
        torch.cuda.empty_cache()
        return results

    # 以下为辅助方法（参数校验、PEFT 管理等）
    def _write_peft_checkpoint(self, local_model, model_name: str):
        """
        将本地模型保存为 PEFT 检查点（如需）。

        Args:
            local_model: 要保存的本地模型实例。
            model_name: 助手模型名称。

        Returns:
            已保存的 PEFT 检查点目录路径。

        Raises:
            FileNotFoundError: 如果无法创建或访问运行时用户目录。
        """

        peft_dir = self._get_peft_dir(model_name)

        # 确保 PEFT 目录存在
        os.makedirs(peft_dir, exist_ok=True)

        # 将本地模型保存为 PEFT 检查点
        local_model.save_pretrained(peft_dir)
        logger.debug(f"Saved PEFT checkpoint to {peft_dir}")

        return peft_dir

    def _validate_session_inputs(
        self,
        task_desc: str,
        single_turn_prompt: str,
        max_new_turns: int,
        local_model: Optional[PreTrainedModel] = None,
        local_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        vllm_base_model: Optional[str] = None,
        assistant_generation_kwargs: Optional[Dict[str, Any]] = None,
        user_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        在开始会话前对所有参数进行基本校验。

        抛出
        ------
        ValueError
            若有任何不满足运行器要求的不变量则抛出。
        """
        if not isinstance(task_desc, str) or not task_desc.strip():
            raise ValueError("`task_desc` must be a non-empty string.")

        if not isinstance(single_turn_prompt, str) or not single_turn_prompt.strip():
            raise ValueError("`single_turn_prompt` must be a non-empty string.")

        if not isinstance(max_new_turns, int) or max_new_turns < 0:
            raise ValueError("`max_new_turns` must be an integer ≥ 0.")

        if (local_model is None) ^ (local_tokenizer is None):
            raise ValueError(
                "Provide *both* `local_model` and `local_tokenizer`, or neither."
            )

        if assistant_generation_kwargs.get("model") is None:
            raise ValueError(
                "`assistant_generation_kwargs` must include a 'model' key."
            )
        if user_generation_kwargs.get("model") is None:
            raise ValueError("`user_generation_kwargs` must include a 'model' key.")

    def _determine_starting_role(self, chat_history: List[Dict[str, str]]) -> str:
        """决定对话应由哪个角色开始（user 或 assistant）。"""
        if chat_history and chat_history[-1]["role"] == "user":
            return "assistant"
        return "user"

    def _get_peft_dir(self, model_name: str) -> str:
        """
        获取 PEFT 检查点目录路径。

        Args:
            model_name: 助手模型名称

        Returns:
            PEFT 检查点目录路径
        """
        run_user_dir = os.environ.get("RUN_USER_DIR")
        return os.path.join(
            run_user_dir, f"{model_name.replace('/', '_')}{PEFT_CHECKPOINT_SUFFIX}"
        )

    def _convert_to_sampling_params(self, generation_kwargs: Dict[str, Any]):
        """
        将 generation kwargs 转换为 vLLM 的 SamplingParams 实例。

        Args:
            generation_kwargs: 生成参数字典

        Returns:
            适用于 vLLM 的 SamplingParams 实例
        """
        from vllm.sampling_params import SamplingParams

        # 过滤出有效的采样参数
        generation_kwargs = copy.deepcopy(generation_kwargs)
        generation_kwargs.pop("model", None)  # 'model' is not a sampling param
        sampling_kwargs = {"max_tokens": 1024}  # 默认 max_tokens
        unmapped_params = []

        for key, value in generation_kwargs.items():
            if key in VALID_VLLM_SAMPLING_PARAMS:
                sampling_kwargs[key] = value
            else:
                unmapped_params.append(key)

        # 记录未映射的参数
        if unmapped_params:
            logger.warning(f"Unmapped VLLM parameters: {unmapped_params}")

        return SamplingParams(**sampling_kwargs)

    def _should_terminate_conversation(self, response: str) -> bool:
        """
        Check if the response contains a termination signal.

        Args:
            response: The response text to check

        Returns:
            True if conversation should terminate, False otherwise
        """
        try:
            return COLLABLLM_TERMINATION_SIGNAL in response
        except Exception as e:
            logger.error(f"Error checking for chat termination: {e}")
            return False

    def _log_response(self, role: str, response: str, prefix: str = "") -> None:
        """Log the response if verbose mode is enabled and on main process."""
        logger.info(
            f"{prefix}[rank {os.environ.get('RANK', 0)}]{role.capitalize()}: {response}"
        )
