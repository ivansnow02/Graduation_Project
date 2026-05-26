"""
collabllm.metric
~~~~~~~~~~~~~~~~
度量与评分接口定义。

该模块提供统一的 metric 基类与若干面向单轮/多轮对话的评分接口，
供训练、评估和奖励计算流程复用。
"""

import abc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from collabllm.prompts import EXTRACT_MULTITURN_COMPLETION_PROMPT
from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)


# 抽象接口
class BaseMetric(abc.ABC):
    """每个度量（metric）必须实现 `score` 方法，并声明其返回的键（keys）。"""

    @abc.abstractmethod
    def score(  # noqa: D401  (imperative mood is OK here)
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """Compute the metric(s) for a prompt–completion pair."""


# 通用驱动
class SingleTurnOrChatMetric:
    """
    一个包装类：可选地将多轮对话日志提取为最终的输出（final completion），
    然后在提取出的文本对上运行具体的度量（metric）。

    *signature* 字符串参考自 DSPy：

        "<extract_type>-><metric_name>"   例如 "document->bert_score"
        "<metric_name>"                   例如 "toxicity"

    • 当包含 `->` 时，先通过 LLM 从完整历史中抽取 `<extract_type>`（文档、答案、策略等）。
    • 随后使用 `<metric_name>` 计算并返回数值评分。
    """

    # 注册表：可通过装饰器注册自定义 metric
    _METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {}

    def __init__(self, signature: str, **llm_kwargs: Any):
        self.extract_type, self.metric_name = self._parse_signature(signature)
        self.llm_kwargs = llm_kwargs
        if "base_url" in self.llm_kwargs:
            self.llm_kwargs["api_base"] = self.llm_kwargs.pop("base_url")

        try:
            metric_cls = self._METRIC_REGISTRY[self.metric_name]
        except KeyError as e:
            raise ValueError(
                f"Metric '{self.metric_name}' is not registered. "
                f"Available: {list(self._METRIC_REGISTRY)}"
            ) from e

        try:
            self.metric: BaseMetric = metric_cls(**self.llm_kwargs)
        except Exception as e:
            self.metric: BaseMetric = metric_cls()

    # 公共 API
    def __call__(  # noqa: D401
        self,
        messages: List[Dict[str, str]],
        single_turn_prompt: str,
        single_turn_completion: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """主入口点：被外部调用以对单轮或多轮对话进行评分。"""
        if self.extract_type:
            completion = self._extract_final_completion(messages, metadata)
        else:
            completion = None

        return self.metric.score(
            single_turn_prompt, single_turn_completion, completion, messages, metadata
        )

    # 辅助方法
    @staticmethod
    def _parse_signature(sig: str) -> Tuple[Optional[str], str]:
        # 解析签名：如果包含 '->' 则分割为 (extract_type, metric_name)，否则只有 metric_name
        return sig.split("->", 1) if "->" in sig else (None, sig)

    def _extract_final_completion(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """调用 LLM 从 `messages` 中提取最终产物（final artefact）。"""
        prefix_msg = (
            "Addtional requirement:\n"
            if metadata and "extraction_requirement" in metadata
            else ""
        )
        prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(
            extract_type=self.extract_type,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            extraction_requirement=prefix_msg
            + metadata.get("extraction_requirement", ""),
        )

        response = (
            litellm.completion(
                **self.llm_kwargs, messages=[{"role": "user", "content": prompt}]
            )
            .choices[0]
            .message.content
        )

        try:
            payload = (
                extract_json(response)
                if isinstance(response, str)
                else response  # Already parsed
            )
            logger.info("Extractor 输出: %s", payload)
            # 验证提取结果包含预期字段 thought 和 final_completion
            if not (
                isinstance(payload, dict)
                and {"thought", "final_completion"} <= payload.keys()
            ):
                raise ValueError("提取结果键不符合预期。")
            return payload["final_completion"]

        except Exception as e:
            logger.error("JSON 提取失败: %s", e)
            raise RuntimeError("无法解析 extractor 的输出；详细信息请查看日志。") from e

    # 注册装饰器
    @classmethod
    def register_metric(cls, name: str):
        """装饰器：将 `metric_cls` 注册到全局注册表中，供签名调用使用。"""

        def _decorator(metric_cls: type[BaseMetric]):
            if name in cls._METRIC_REGISTRY:
                logger.warning(
                    f"将用 {metric_cls.__name__} 覆盖已存在的 metric '{name}'。"
                )
            cls._METRIC_REGISTRY[name] = metric_cls
            return metric_cls

        return _decorator

# 导入并注册标准的 metric 实现
try:
    from collabllm.metrics.teaching_quality import TeachingQualityMetric

    SingleTurnOrChatMetric.register_metric("teaching_quality")(TeachingQualityMetric)
except ImportError:
    pass
