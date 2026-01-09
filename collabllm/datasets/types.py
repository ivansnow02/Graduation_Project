# 1. 定义对话轮次类
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class DialogueTurn:
    role: str
    content: str


# 2. 定义标注信息类
@dataclass
class Annotation:
    speaker: str
    utterance: str
    discipline: str
    discipline_transfer: str
    student_cognition_state: str
    cognitive_level: str
    # 下面这些字段在某些标注中可能为空，所以设为 Optional，默认为空字符串
    teacher_intent: str = ""
    teaching_strategy: str = ""
    teacher_guidance_level: str = ""


# 3. 定义主数据结构类
@dataclass
class TeachingSession:
    student_id: str
    student_type: str
    scenario: str
    topic_id: str
    topic_text: str
    repeat_id: str
    dialogue: List[DialogueTurn]
    annotations: List[Annotation]
    quality_score: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        工厂方法：将字典转换为对象，自动处理嵌套列表
        """
        # 处理 dialogue 列表
        dialogue_objs = [DialogueTurn(**turn) for turn in data.get("dialogue", [])]

        # 处理 annotations 列表
        # 注意：这里做了一些容错处理，防止 json 里的字段比类定义的字段多导致报错
        annotation_objs = []
        for ann in data.get("annotations", []):
            # 过滤掉不在 Annotation 类定义中的额外字段（如果有的话），防止报错
            # 如果你确定 JSON 字段完全匹配，可以直接用 Annotation(**ann)
            valid_keys = Annotation.__annotations__.keys()
            filtered_ann = {k: v for k, v in ann.items() if k in valid_keys}
            annotation_objs.append(Annotation(**filtered_ann))

        return cls(
            student_id=data.get("student_id", ""),
            student_type=data.get("student_type", ""),
            scenario=data.get("scenario", ""),
            topic_id=data.get("topic_id", ""),
            topic_text=data.get("topic_text", ""),
            repeat_id=data.get("repeat_id", ""),
            dialogue=dialogue_objs,
            annotations=annotation_objs,
        )

    def to_dict(self):
        """将对象转回字典，方便再次存为 JSON"""
        return asdict(self)

    def check_consistency(self) -> bool:
        """清洗示例：检查对话和标注的一致性"""
        # 注意：你的数据中 dialogue 长度是 12，annotations 长度是 14
        # 因为最后一段话被拆分成了三个标注。
        # 这里只是演示如何访问数据
        print(
            f"ID: {self.student_id} | 对话数: {len(self.dialogue)} | 标注数: {len(self.annotations)}"
        )
        return len(self.dialogue) == len(self.annotations)

    def clean_empty_annotations(self):
        """清洗示例：移除内容为空的标注"""
        original_count = len(self.annotations)
        self.annotations = [ann for ann in self.annotations if ann.utterance.strip()]
        if len(self.annotations) < original_count:
            print(f"已移除 {original_count - len(self.annotations)} 条空标注")

    def get_strategy_stats(self):
        """分析示例：统计用到的教学策略"""
        strategies = [
            ann.teaching_strategy for ann in self.annotations if ann.teaching_strategy
        ]
        return strategies
