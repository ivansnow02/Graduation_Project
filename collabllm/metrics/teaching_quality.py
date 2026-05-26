from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, List, Optional

import litellm
from collabllm.metric import BaseMetric
from collabllm.datasets.types import Annotation
from collabllm.utils.metrics import calculate_turn_metrics
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# Prompt 规则
ANNOTATION_HEADER = """
你是一位严苛且专业的教育认知标注专家。请根据下面一段完整的教学对话，从第一轮开始逐轮提取以下9项教学信息，并输出为结构化 JSON 格式（列表形式）。
请不要跳过任何一轮。你需要结合全局上下文来准确判断教师的引导深度。
"""

ANNOTATION_SPEC = """
## 【需要标注的字段】：
- speaker：发言者（"教师"或"学生"）
- utterance：原始发言文本，**不能进行任何修改**
- teacher_intent：教师发言中体现的教学目的（如"引出概念"、"引导推理"等），学生轮为空。
- teaching_strategy：教师采用的策略。
  裁判铁律：如果教师仅仅是把学生的话改成问号复读一遍，没有任何新信息或引导方向，请标注为"无效复读"！
- discipline：该轮涉及的学科。
- discipline_transfer：若当前轮相较上轮出现新的学科，引导学科间联系，请填"是"，否则填"否"。
- student_cognition_state：仅学生轮填写。
- teacher_guidance_level：分为3级：
  - L1：封闭性问题（是/否、定义型）或 无效的纯复读反问。
  - L2：解释/理解型问题，或提供了一定背景知识的提问。
  - L3：迁移、推理、综合型问题（提供了知识脚手架并引发深度思考）。
- cognitive_level：根据 Bloom 分类填写。
"""

ANNOTATION_OUTPUT_REQ = """
##【请输出如下结构化标注】：（严格JSON格式，务必使用双引号，按对话顺序输出包含所有轮次的列表）
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
    """教学质量度量。"""

    def __init__(self, **llm_kwargs):
        self.llm_kwargs = llm_kwargs
        if "temperature" not in self.llm_kwargs:
            self.llm_kwargs["temperature"] = 0.0
        if "max_tokens" not in self.llm_kwargs:
            self.llm_kwargs["max_tokens"] = 4096

    # 全局缓存
    _CACHE: Dict[str, List[Annotation]] = {}

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not messages:
            logger.warning(
                "TeachingQualityMetric received empty messages. Returning 0."
            )
            return 0.0

        # 只保留对话消息
        filtered_msgs = [
            m
            for m in messages
            if m["role"].lower() in ("user", "assistant", "student", "teacher")
        ]

        # 直接进行全局标注
        annotations = self._annotate_full_dialogue(filtered_msgs)

        if not annotations:
            return 0.0

        # 倒序查找最后一个教师轮次
        last_teacher_idx = -1
        for i in range(len(annotations) - 1, -1, -1):
            if (
                annotations[i].speaker == "教师"
                or annotations[i].is_teacher_annotation()
            ):
                last_teacher_idx = i
                break

        if last_teacher_idx == -1:
            logger.warning("No Teacher annotation found in messages. Returning 0.0.")
            return 0.0

        last_ann = annotations[last_teacher_idx]
        history_anns = annotations[:last_teacher_idx]

        # 将带有上下文历史的标注丢给打分器
        scores = calculate_turn_metrics(last_ann, history_annotations=history_anns)

        return scores["total_score"]

    def _annotate_full_dialogue(
        self, messages: List[Dict[str, str]]
    ) -> List[Annotation]:
        """将完整的对话历史发给大模型进行一次性标注"""

        # 生成基于完整上下文的唯一缓存哈希值 (使用 MD5 防止 Key 过长)
        key_content = "||".join(
            [f"{msg.get('role', '')}:{msg.get('content', '')}" for msg in messages]
        )
        cache_key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        # 命中缓存直接返回
        if cache_key in self._CACHE:
            return self._CACHE[cache_key]

        # 未命中，拼接全局 Prompt
        dialogue_text = "##【对话内容】：\n"
        for msg in messages:
            role = "教师" if msg["role"].lower() in ("assistant", "teacher") else "学生"
            dialogue_text += f"{role}：{msg['content']}\n"

        prompt_text = (
            ANNOTATION_HEADER + ANNOTATION_SPEC + dialogue_text + ANNOTATION_OUTPUT_REQ
        ).strip()

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
                annotations = [Annotation.from_dict(item) for item in json_obj]
                self._CACHE[cache_key] = annotations
                return annotations
            else:
                logger.warning(f"Annotation output is not a list: {json_obj}")
                return []

        except Exception as e:
            logger.error(f"Error annotating dialogue: {e}")
            return []
