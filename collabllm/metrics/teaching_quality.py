from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import litellm
from collabllm.metric import BaseMetric
from collabllm.datasets.types import Annotation
from collabllm.utils.metrics import calculate_turn_metrics
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# Annotation Prompt Template (Ported from scripts/benchmark/annotation.py)
ANNOTATION_HEADER = """
你是一位教育认知标注专家，请根据下面一段教学对话，逐轮提取以下9项教学信息，并输出为结构化 JSON 格式（列表形式）。
每一轮包含教师或学生的一个发言。请不要跳过任何一轮。
"""
ANNOTATION_SPEC = """
## 【需要标注的字段】：
- speaker：发言者（"教师"或"学生"）
- utterance：原始发言文本，**不能进行任何修改**
- teacher_intent：教师发言中体现的教学目的，有以下五种："引出概念"、"检测理解"、"引导推理"、"引发迁移"、"总结提升"，**学生轮为空字符串**
- teaching_strategy：教师采用的策略，如"追问"、"提示"、"类比"、"情境设问"、"拆解问题"、"鼓励回应"、"正误反馈"等，**学生轮为空字符串**
- discipline：该轮涉及的学科，如"地理"、"生物"、"物理"、"历史"，多个学科请用逗号分隔
- discipline_transfer：若当前轮相较上轮出现新的学科，引导学科间联系，请填"是"，否则填"否"
- student_cognition_state：**仅学生轮填写**，有以下几种："清晰理解"、"模糊理解"、"表达困难"、"答非所问"、"错误回答"、"高阶思考"；**教师轮为空字符串**
- teacher_guidance_level：**仅教师轮填写**，分为3级：
  - L1：封闭性问题（是/否、定义型）
  - L2：解释/理解型问题
  - L3：迁移、推理、综合型问题
  只能标注L1，L2，L3三种
- cognitive_level：请根据 Bloom 分类，选择以下之一：
  - 记忆（Remember）
  - 理解（Understand）
  - 应用（Apply）
  - 分析（Analyze）
  - 评价（Evaluate）
  - 创造（Create）
"""
ANNOTATION_OUTPUT_REQ = """
##【请输出如下结构化标注】：（严格JSON格式，请务必使用双引号）
[
  {
    "speaker": "教师",
    "utterance": "...",
    "teacher_intent": "...",
    "teaching_strategy": "...",
    "discipline": "...",
    "discipline_transfer": "...",
    "student_cognition_state": "",
    "teacher_guidance_level": "...",
    "cognitive_level": "..."
  },
  {
    "speaker": "学生",
    "utterance": "...",
    "teacher_intent": "",
    "teaching_strategy": "",
    "discipline": "...",
    "discipline_transfer": "...",
    "student_cognition_state": "...",
    "teacher_guidance_level": "",
    "cognitive_level": "..."
  }
]
请从第一轮开始逐轮标注，直到对话结束。
"""


class TeachingQualityMetric(BaseMetric):
    """
    Computes teaching quality metrics by first annotating the conversation using an LLM,
    and then calculating scores based on the 7-dimensional framework in `collabllm.utils.metrics`.
    """

    def __init__(self, **llm_kwargs):
        """
        Args:
            **llm_kwargs: Arguments passed to litellm.completion (e.g. model, temperature).
        """
        self.llm_kwargs = llm_kwargs
        # Set default values if not provided
        if "temperature" not in self.llm_kwargs:
            self.llm_kwargs["temperature"] = 0.0  # Deterministic for annotation
        if "max_tokens" not in self.llm_kwargs:
            self.llm_kwargs["max_tokens"] = 4096

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Annotate the full `messages` history and compute the teaching quality score.
        """
        if not messages:
            logger.warning(
                "TeachingQualityMetric received empty messages. Returning 0."
            )
            return 0.0

        # 1. Annotate the dialogue
        annotations = self._annotate_dialogue(messages)

        # 2. Calculate Metrics (Incremental Reward)
        # We want to score the LAST turn (the candidate response).
        # Constraint: The last message in `messages` MUST be the assistant response we are evaluating.
        # `multiturn_aware_reward` appends the candidate to messages before calling score.

        if not annotations:
            return 0.0

        # Identify the annotation corresponding to the last assistant turn.
        # Find the last teacher annotation
        teacher_anns = [ann for ann in annotations if ann.is_teacher_annotation()]
        if not teacher_anns:
            logger.warning("No Teacher annotation found in messages. Returning 0.0.")
            return 0.0

        last_ann = teacher_anns[-1]
        last_teacher_idx = annotations.index(last_ann)
        history_anns = annotations[:last_teacher_idx]
        scores = calculate_turn_metrics(last_ann, history_annotations=history_anns)

        return scores["total_score"]

    def _annotate_dialogue(self, messages: List[Dict[str, str]]) -> List[Annotation]:
        """
        Annotate the dialogue by processing it in pairs (User, Assistant) or chunks.
        Replicates the logic from `scripts/benchmark/annotation.py`.
        """

        # Process in pairs: (User, Assistant) usually.
        # But messages might be [User, Assistant, User, Assistant...]
        # annotation.py does: for i in range(0, len(turns) - 1, 2): pair = turns[i : i + 2]

        # If messages have system prompt? The caller (multiturn_aware_reward) strips it.
        # But let's filter just in case.
        filtered_msgs = [
            m
            for m in messages
            if m["role"]
            in ("user", "assistant", "student", "teacher", "User", "Assistant")
        ]

        pairs = []
        for i in range(0, len(filtered_msgs), 2):
            # Take slices of 2
            pair = filtered_msgs[i : i + 2]
            if not pair:
                break
            pairs.append(pair)

        # Parallel annotation could be faster, but let's stick to sequential or use ThreadPool if provided
        # BaseMetric doesn't control concurrency, but we can usage internal ThreadPool if we want.
        # Given this is inside a reward function that is already threaded?
        # `multiturn_aware_reward` runs `_score_one_metric` in a ThreadPool.
        # So we are already inside a thread. Better run sequentially to avoids thread-explosion
        # or rate-limit issues, unless we are sure.
        # However, annotating a long conversation sequentially is slow.
        # Let's try sequential for stability first.

        all_pair_annotations = []
        for pair in pairs:
            pair_anns = self._annotate_pair(pair)
            if pair_anns:
                all_pair_annotations.extend(pair_anns)

        return all_pair_annotations

    # Class-level cache to store annotations for dialogue pairs
    # Key: hash of the pair content
    # Value: List[Annotation]
    _CACHE: Dict[str, List[Annotation]] = {}

    def _annotate_pair(self, pair: List[Dict[str, str]]) -> List[Annotation]:
        """Annotate a single pair of turns with caching."""
        # 1. Generate a cache key based on content
        # We use a tuple of (role, content) for each message to ensure uniqueness
        key_items = []
        for msg in pair:
            key_items.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
        cache_key = "||".join(key_items)

        # 2. Check cache
        if cache_key in self._CACHE:
            # Return a deep copy to avoid mutation issues if any
            # (Annotation is a Pydantic model or similar, usually immutable-ish, but let's be safe)
            # Actually, just returning the list is fine if we don't modify it.
            return self._CACHE[cache_key]

        # 3. If miss, Call LLM
        prompt_text = self._build_prompt(pair)

        try:
            response = (
                litellm.completion(
                    messages=[{"role": "user", "content": prompt_text}],
                    **self.llm_kwargs,
                )
                .choices[0]
                .message.content
            )

            json_obj = extract_json(response)

            if isinstance(json_obj, list):
                # Convert dicts to Annotation objects
                annotations = [Annotation.from_dict(item) for item in json_obj]
                # 4. Update cache
                self._CACHE[cache_key] = annotations
                return annotations
            else:
                logger.warning(f"Annotation output is not a list: {json_obj}")
                return []

        except Exception as e:
            logger.error(f"Error annotating pair: {e}")
            return []

    def _build_prompt(self, pair: List[Dict[str, str]]) -> str:
        """Parameters the prompt for the annotator."""
        dialogue_text = "##【对话内容】：\n"
        for msg in pair:
            role = "教师" if msg["role"] in ("assistant", "teacher") else "学生"
            dialogue_text += f"{role}：{msg['content']}\n"

        return (
            ANNOTATION_HEADER + ANNOTATION_SPEC + dialogue_text + ANNOTATION_OUTPUT_REQ
        ).strip()


# Register directly if needed, but the main registry is in metric.py
