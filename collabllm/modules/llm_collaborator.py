"""
collabllm.modules.llm_collaborator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM 协作端封装。

该模块负责调用模型生成协作回复，并对返回结果进行解析与重试。
"""

from typing import List
import logging
import litellm

from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json
from collabllm.prompts import PROACT_MODEL_PROMPT

logger = logging.getLogger(__name__)


class LLMCollaborator(object):
    registered_prompts = {"none": None, "proact": PROACT_MODEL_PROMPT}

    def __init__(self, method="none", num_retries=10, **llm_kwargs):
        """初始化 `LLMCollaborator`。

        Args:
            method: 提示方法，必须在 `registered_prompts` 中注册。
            num_retries: 请求失败时的重试次数。
            llm_kwargs: 传递给 LLM 的额外参数。
        """
        super().__init__()
        self.method = method
        assert method in self.registered_prompts, (
            f"Prompting method {method} not registered. Available methods: {list(self.registered_prompts.keys())}"
        )

        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}

        # 支持使用 base_url 字段来兼容 vLLM/OpenAI 风格的后端配置
        if "base_url" in self.llm_kwargs:
            self.llm_kwargs["api_base"] = self.llm_kwargs.pop("base_url")

    def __call__(self, messages: List[dict], **kwargs):
        """模型的调用接口（前向流程）。

        Args:
            messages (List[dict]): 消息列表，最后一条应为用户消息。

        Returns:
            str: 模型生成的回复文本（已 strip）。
        """
        assert messages[-1]["role"] == "user"

        if self.method == "none":
            if len(messages) and messages[0]["role"] == "system":
                logger.info("检测到 system 消息。")
        else:
            kwargs = {}
            prompt = PROACT_MODEL_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                max_new_tokens=self.llm_kwargs.get("max_new_tokens", 1024),
                additional_info=kwargs.get("additional_info", ""),
            )
            messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            full_response = (
                litellm.completion(
                    **self.llm_kwargs,
                    messages=messages,
                    num_retries=self.num_retries,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                .choices[0]
                .message.content
            )

            try:
                if isinstance(full_response, str) and not (self.method == "none"):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[LLMCollaborator] JSON 提取错误: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {"current_problem", "thought", "response"}.issubset(keys):
                    response = full_response.pop("response")
                    break
                else:
                    logger.error(
                        f"[LLMCollaborator] 返回键 {keys} 与预期不匹配，正在重试..."
                    )
                    continue
            else:
                response = full_response
                break

        return response.strip()
