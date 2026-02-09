from typing import List
import logging
import litellm

from collabllm.prompts import USER_SIMULATOR_PROMPT, COLLABLLM_TERMINATION_SIGNAL
from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)


class UserSimulator(object):
    def __init__(
        self,
        task_desc="",
        single_turn_prompt="",
        num_retries=10,
        prompt_template: str = None,
        require_json: bool = True,
        **llm_kwargs,
    ):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_desc = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.num_retries = num_retries
        self.prompt_template = prompt_template
        self.require_json = require_json

        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 1024, **llm_kwargs}

        # Support base_url for vLLM/OpenAI compatible backends
        if "base_url" in self.llm_kwargs:
            self.llm_kwargs["api_base"] = self.llm_kwargs.pop("base_url")

        assert "model" in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def __call__(self, messages: List[dict]):
        template = (
            self.prompt_template if self.prompt_template else USER_SIMULATOR_PROMPT
        )

        prompt = template.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            terminal_signal=COLLABLLM_TERMINATION_SIGNAL,
        )
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            full_response = (
                litellm.completion(
                    **self.llm_kwargs,
                    messages=messages,
                    num_retries=self.num_retries,
                )
                .choices[0]
                .message.content
            )

            if not self.require_json:
                return full_response.strip()

            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[UserSimulator] Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {"current_answer", "thought", "response"}.issubset(keys):
                    response = full_response.pop("response")
                    break
                else:
                    logger.error(
                        f"[UserSimulator] Keys {keys} do not match expected keys. Retrying..."
                    )
                    continue

        return response.strip()
