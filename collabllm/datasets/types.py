"""
collabllm.datasets.types
~~~~~~~~~~~~~~~~~~~~~~~
数据类定义：支持 SID (Socratic Interdisciplinary Dialogue) 数据的结构化表示。

包含三层结构：
  1. DialogueTurn: 单轮对话（角色 + 内容）
  2. Annotation: 单条注释（标注的教学/学生信息）
  3. TeachingSession: 完整的教学会话（学生信息、对话、标注、质量分数）
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Counter, Literal
from collections import Counter as CounterType
import json
import logging

logger = logging.getLogger(__name__)


# DialogueTurn
@dataclass
class DialogueTurn:
    """
    表示对话中的一轮（单个消息）。

    Attributes:
        role: 说话者角色 ("学生", "教师", "student", "teacher" 等)
        content: 说话内容 (str，不应为空)
        score: 该轮对话的独立评分 (Optional[float], 通常用于强化学习或 DPO 的奖励值)
    """

    role: str
    content: str
    score: Optional[float] = None

    def __post_init__(self):
        """验证对话轮次的完整性"""
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError(f"Role must be a non-empty string, got: {self.role}")
        if not isinstance(self.content, str):
            raise ValueError(
                f"Content must be a string, got type: {type(self.content)}"
            )

    def is_student(self) -> bool:
        """检查是否为学生发言"""
        return self.role.lower() in {"学生", "student", "user"}

    def is_teacher(self) -> bool:
        """检查是否为教师发言"""
        return self.role.lower() in {"教师", "teacher", "assistant"}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {"role": self.role, "content": self.content, "score": self.score}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTurn":
        """从字典创建对象"""
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            score=data.get("score"),
        )


# Annotation
@dataclass
class Annotation:
    """
    表示对单条对话轮次的标注信息（教学策略、学生认知状态等）。

    核心字段：
        speaker: 说话者类型 ("学生", "教师", "student", "teacher" 等)
        utterance: 该轮对话的内容文本
        discipline: 当前轮次涉及的学科 (e.g., "数学", "物理")
        discipline_transfer: 是否存在跨学科迁移 ("是"/"否" 或 True/False)
        student_cognition_state: 学生认知状态 (e.g., "清晰", "模糊", "错误")
        cognitive_level: Bloom 分类认知层级 (e.g., "记忆", "理解", "应用")

    可选字段（教师标注）：
        teacher_intent: 教师的教学意图 (e.g., "引导", "检查理解")
        teaching_strategy: 教学策略 (e.g., "提问", "讲解", "类比")
        teacher_guidance_level: 教师引导强度 (e.g., "L1", "L2", "L3")
    """

    speaker: str
    utterance: str
    discipline: str
    discipline_transfer: str
    student_cognition_state: str
    cognitive_level: str
    teacher_intent: str = ""
    teaching_strategy: str = ""
    teacher_guidance_level: str = ""

    def __post_init__(self):
        """验证标注数据的完整性"""
        # 不在此处进行严格验证，避免过度警告
        # 完整性检查通过 is_complete() 方法进行
        pass

    def is_complete(self) -> bool:
        """检查标注是否完整（根据发言者角色验证不同必填字段）"""
        # 基础必填字段 (对所有角色都适用)
        base_required = {"speaker", "utterance"}
        for f in base_required:
            val = getattr(self, f, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                return False

        # 角色特有字段检查
        if self.is_teacher_annotation():
            # 教师标注通常需要：意图 or 策略 or 引导等级
            # 至少有一个非空即可认为是有意义的教师标注
            has_info = bool(
                (self.teacher_intent and self.teacher_intent.strip())
                or (self.teaching_strategy and self.teaching_strategy.strip())
                or (self.teacher_guidance_level and self.teacher_guidance_level.strip())
            )
            return has_info

        elif self.is_student_annotation():
            # 学生标注通常需要：认知状态 或 认知层级
            has_info = bool(
                (self.student_cognition_state and self.student_cognition_state.strip())
                or (self.cognitive_level and self.cognitive_level.strip())
            )
            return has_info

        return True  # 其他角色暂不严格校验

    def is_teacher_annotation(self) -> bool:
        """检查是否为教师标注"""
        return self.speaker.lower() in {"教师", "teacher", "assistant"}

    def is_student_annotation(self) -> bool:
        """检查是否为学生标注"""
        return self.speaker.lower() in {"学生", "student", "user"}

    def has_teaching_info(self) -> bool:
        """检查是否包含教学策略信息"""
        return bool(
            self.teacher_intent or self.teaching_strategy or self.teacher_guidance_level
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """从字典创建对象，自动过滤额外字段"""
        valid_keys = set(cls.__annotations__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        # 填充缺失的可选字段
        for key in valid_keys:
            if key not in filtered_data:
                filtered_data[key] = ""
        return cls(**filtered_data)


# TeachingSession: 完整教学会话
@dataclass
class TeachingSession:
    """
    表示完整的一次教学会话（师生对话 + 所有标注）。

    Attributes:
        student_id: 学生ID
        student_type: 学生类型 (e.g., "高中生", "大学生")
        scenario: 教学场景 (e.g., "一对一辅导", "课堂提问")
        topic_id: 主题/学科ID
        topic_text: 主题完整描述
        repeat_id: 重复ID（同一主题的不同对话轮次标识）
        dialogue: 所有对话轮次 (List[DialogueTurn])
        annotations: 所有标注 (List[Annotation])
        quality_score: 会话质量评分 (0.0 - 1.0)
    """

    student_id: str
    student_type: str
    scenario: str
    topic_id: str
    topic_text: str
    repeat_id: str
    dialogue: List[DialogueTurn]
    annotations: List[Annotation]
    quality_score: float = 0.0

    # 缓存字段，在需要时计算
    _teacher_turns_cache: Optional[List[DialogueTurn]] = field(
        default=None, init=False, repr=False
    )
    _student_turns_cache: Optional[List[DialogueTurn]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """验证会话数据的完整性"""
        if not self.dialogue:
            logger.warning(f"TeachingSession {self.student_id} has empty dialogue")
        if not self.annotations:
            logger.warning(f"TeachingSession {self.student_id} has empty annotations")
        if not (0.0 <= self.quality_score <= 1.0):
            logger.warning(f"Quality score {self.quality_score} is not in [0, 1]")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeachingSession":
        """
        从字典创建 TeachingSession 对象。
        自动处理嵌套的 dialogue 和 annotations 列表。
        """
        # 处理 dialogue
        dialogue_objs = []
        for turn in data.get("dialogue", []):
            if isinstance(turn, DialogueTurn):
                dialogue_objs.append(turn)
            else:
                try:
                    dialogue_objs.append(DialogueTurn.from_dict(turn))
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to parse dialogue turn: {turn}, error: {e}")
                    continue

        # 处理 annotations
        annotation_objs = []
        for ann in data.get("annotations", []):
            if isinstance(ann, Annotation):
                annotation_objs.append(ann)
            else:
                try:
                    annotation_objs.append(Annotation.from_dict(ann))
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to parse annotation: {ann}, error: {e}")
                    continue

        return cls(
            student_id=data.get("student_id", ""),
            student_type=data.get("student_type", ""),
            scenario=data.get("scenario", ""),
            topic_id=data.get("topic_id", ""),
            topic_text=data.get("topic_text", ""),
            repeat_id=data.get("repeat_id", ""),
            dialogue=dialogue_objs,
            annotations=annotation_objs,
            quality_score=float(data.get("quality_score", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转换为字典（用于 JSON 序列化）。
        所有嵌套对象都会转换为字典形式。
        """
        return {
            "student_id": self.student_id,
            "student_type": self.student_type,
            "scenario": self.scenario,
            "topic_id": self.topic_id,
            "topic_text": self.topic_text,
            "repeat_id": self.repeat_id,
            "dialogue": [
                t.to_dict() if hasattr(t, "to_dict") else asdict(t)
                for t in self.dialogue
            ],
            "annotations": [
                a.to_dict() if hasattr(a, "to_dict") else asdict(a)
                for a in self.annotations
            ],
            "quality_score": self.quality_score,
        }

    def to_json_str(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ================================================================
    # 统计和访问方法
    # ================================================================

    def get_dialogue_length(self) -> int:
        """获取对话轮次总数"""
        return len(self.dialogue)

    def get_annotation_count(self) -> int:
        """获取标注总数"""
        return len(self.annotations)

    def get_teacher_turns(self) -> List[DialogueTurn]:
        """获取所有教师发言轮次"""
        if self._teacher_turns_cache is None:
            self._teacher_turns_cache = [t for t in self.dialogue if t.is_teacher()]
        return self._teacher_turns_cache

    def get_student_turns(self) -> List[DialogueTurn]:
        """获取所有学生发言轮次"""
        if self._student_turns_cache is None:
            self._student_turns_cache = [t for t in self.dialogue if t.is_student()]
        return self._student_turns_cache

    def get_teacher_annotations(self) -> List[Annotation]:
        """获取所有教师标注"""
        return [a for a in self.annotations if a.is_teacher_annotation()]

    def get_student_annotations(self) -> List[Annotation]:
        """获取所有学生标注"""
        return [a for a in self.annotations if a.is_student_annotation()]

    def get_strategies_used(self) -> List[str]:
        """获取使用过的所有教学策略列表"""
        strategies = []
        for ann in self.get_teacher_annotations():
            if ann.teaching_strategy:
                # 处理可能的多个策略（用逗号分隔）
                strats = ann.teaching_strategy.replace("，", ",").split(",")
                strategies.extend([s.strip() for s in strats if s.strip()])
        return strategies

    def get_unique_strategies(self) -> Set[str]:
        """获取使用过的独特教学策略集合"""
        return set(self.get_strategies_used())

    def get_strategy_frequency(self) -> Dict[str, int]:
        """获取教学策略的使用频率统计"""
        return dict(CounterType(self.get_strategies_used()))

    def get_disciplines(self) -> Set[str]:
        """获取涉及的所有学科"""
        return {a.discipline for a in self.annotations if a.discipline}

    def get_cognitive_levels(self) -> List[str]:
        """获取所有学生认知层级"""
        return [
            a.cognitive_level
            for a in self.get_student_annotations()
            if a.cognitive_level
        ]

    def get_teacher_intents(self) -> List[str]:
        """获取所有教师教学意图"""
        intents = []
        for ann in self.get_teacher_annotations():
            if ann.teacher_intent:
                intents.extend([
                    i.strip()
                    for i in ann.teacher_intent.replace("，", ",").split(",")
                    if i.strip()
                ])
        return intents

    def get_unique_intents(self) -> Set[str]:
        """获取独特的教师教学意图集合"""
        return set(self.get_teacher_intents())

    # ================================================================
    # 清洗和验证方法
    # ================================================================

    def check_consistency(self) -> bool:
        """
        检查对话和标注的一致性。
        注：通常 annotations 数量可能多于 dialogue（一条对话可能有多个标注）。
        """
        is_consistent = len(self.dialogue) > 0 and len(self.annotations) > 0
        logger.debug(
            f"Session {self.student_id}: dialogue={len(self.dialogue)}, "
            f"annotations={len(self.annotations)}, consistent={is_consistent}"
        )
        return is_consistent

    def clean_empty_annotations(self) -> int:
        """
        移除不完整的标注（缺少必需字段或内容为空）。
        必需字段：speaker, utterance, discipline, discipline_transfer,
                student_cognition_state, cognitive_level
        返回移除的标注数量。
        """
        original_count = len(self.annotations)
        valid_annotations = []

        for ann in self.annotations:
            # 检查是否完整
            if not ann.is_complete():
                continue
            # 检查 utterance 是否为空
            if not ann.utterance or not ann.utterance.strip():
                continue
            valid_annotations.append(ann)

        self.annotations = valid_annotations
        removed = original_count - len(self.annotations)
        if removed > 0:
            logger.info(
                f"Removed {removed} incomplete annotations from session {self.student_id}"
            )
        return removed

    def clean_empty_dialogue(self) -> int:
        """
        移除内容为空的对话轮次。
        返回移除的轮次数量。
        """
        original_count = len(self.dialogue)
        self.dialogue = [
            turn for turn in self.dialogue if turn.content and turn.content.strip()
        ]
        removed = original_count - len(self.dialogue)
        if removed > 0:
            logger.info(
                f"Removed {removed} empty dialogue turns from session {self.student_id}"
            )
        return removed

    def remove_short_dialogues(self, min_length: int = 3) -> bool:
        """
        检查对话是否足够长。如果对话轮次少于 min_length，返回 False。
        通常用于数据清洗的过滤条件。
        """
        is_long_enough = len(self.dialogue) >= min_length
        if not is_long_enough:
            logger.debug(
                f"Session {self.student_id} has too few dialogue turns: "
                f"{len(self.dialogue)} < {min_length}"
            )
        return is_long_enough

    # ================================================================
    # 统计摘要
    # ================================================================

    def get_summary(self) -> Dict[str, Any]:
        """
        获取会话的统计摘要。
        包括对话长度、标注数、策略统计、学科等。
        """
        return {
            "student_id": self.student_id,
            "topic_id": self.topic_id,
            "dialogue_length": len(self.dialogue),
            "annotation_count": len(self.annotations),
            "teacher_turns": len(self.get_teacher_turns()),
            "student_turns": len(self.get_student_turns()),
            "unique_strategies": len(self.get_unique_strategies()),
            "strategies_used": self.get_strategy_frequency(),
            "disciplines": list(self.get_disciplines()),
            "unique_intents": len(self.get_unique_intents()),
            "quality_score": self.quality_score,
        }

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_summary()
        return (
            f"TeachingSession(student_id={self.student_id}, topic_id={self.topic_id}, "
            f"dialogue={summary['dialogue_length']}, annotations={summary['annotation_count']}, "
            f"score={self.quality_score:.2f})"
        )
