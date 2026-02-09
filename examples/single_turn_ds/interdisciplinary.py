# interdisciplinary.py
from __future__ import annotations

import json
from typing import Dict, List, Any
from datasets import Dataset
from collabllm.datasets.single_turn import SingleTurnDataset

# Definitions from scripts/benchmark/multi_dialogue.py
STUDENT_TYPES = {
    "全优型": "你是一位全优型学生，逻辑清晰，善于跨学科推理。",
    "知识掌握不足": "你是一位基础薄弱的学生，对概念掌握不牢，经常需要老师解释基础词汇。",
    "学习渴望低": "你是一位兴趣不高的学生，回答简短，偶尔会表现出不耐烦。",
}

SCENARIOS_TEXT = """
（1）学生不理解问题的含义
（2）学生尝试回答但有部分错误
（3）学生进行了一个类比猜测
（4）学生完全不知道，请求提示
（5）学生回答正确并尝试延伸
""".strip()


class InterdisciplinaryDataset(SingleTurnDataset):
    """
    Adapter for Interdisciplinary Topic Dataset.
    Source: data/interdisciplinary_topic.json

    Mapping Strategy:
    - Expands each topic into 3 examples (one for each Student Type).
    - Embeds Persona and Scenario instructions into the prompt.
    """

    def __init__(
        self,
        data_path: str = "data/interdisciplinary_topic.json",
        *,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed = self._preprocess(raw_data)
        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    @staticmethod
    def _preprocess(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []

        for i, entry in enumerate(raw_data):
            topic_text = entry.get("topic", "")
            if not topic_text:
                continue

            # For each topic, create one example per Student Type
            for s_type, s_desc in STUDENT_TYPES.items():
                # Construct a comprehensive prompt for the User Simulator
                # This prompt guides the simulator to:
                # 1. Ask the initial question (based on topic).
                # 2. Maintain the specific student persona during the dialogue.
                # 3. Randomly vary states (scenarios) as instructed.

                formatted_prompt = f"""
【角色设定】
身份：中学生
类型：{s_type}
描述：{s_desc}

【状态指令】
在后续与教师的对话中，请在每次回答前随机从以下状态中选择一种来模拟真实反应：
{SCENARIOS_TEXT}

【初始任务】
请根据下面的课程内容，作为一名{s_type}的学生，提出一个你真正感到困惑或好奇的问题。
要求：
1. 只需要输出问题本身，不要输出“学生：”或角色的前缀。
2. 问题要具体，不要太宽泛，符合你的角色设定。

【课程内容】
{topic_text}
""".strip()

                processed.append(
                    {
                        "prompt": formatted_prompt,
                        "completion": "",
                        "topic_id": entry.get("id", f"topic_{i + 1}"),
                        "original_topic": topic_text,
                        "student_type": s_type,
                        "student_desc": s_desc,
                    }
                )

        return processed
