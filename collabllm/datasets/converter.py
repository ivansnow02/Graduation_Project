"""
collabllm.datasets.converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
将 TeachingSession 转换为 CollabLLM 标准格式的工具模块。

支持的转换格式：
  1. 嵌套格式 (Nested Structure): 用于 MultiturnDataset 加载
  2. 扁平格式 (Flat List): 备选的扁平化表示
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
from collections import defaultdict

from collabllm.datasets.types import TeachingSession, DialogueTurn, Annotation

logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数
# ============================================================================


def normalize_role(role: str) -> str:
    """
    将角色标准化为 "user" 或 "assistant"。

    Args:
        role: 原始角色字符串 ("学生", "student", "教师", "teacher" 等)

    Returns:
        标准化后的角色 ("user" 或 "assistant")
    """
    role_lower = role.lower()
    if role_lower in {"学生", "student", "user"}:
        return "user"
    elif role_lower in {"教师", "teacher", "assistant"}:
        return "assistant"
    else:
        logger.warning(f"Unknown role: {role}, defaulting to 'user'")
        return "user"


def get_content_hash(messages: List[Dict[str, str]], length: int = 8) -> str:
    """
    计算消息列表的 MD5 哈希。

    Args:
        messages: 消息列表 (List of {role, content})
        length: 哈希截取长度 (默认 8)

    Returns:
        MD5 哈希字符串
    """
    import json

    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:length]


# ============================================================================
# 核心转换函数
# ============================================================================


def convert_session_to_nested(
    session: TeachingSession,
    use_quality_score: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将单个 TeachingSession 转换为 CollabLLM 嵌套格式。

    嵌套格式结构：
    {
        "conv_id": <唯一ID>,
        "single_turn_prompt": <原始任务描述>,
        "single_turn_completion": <原始单轮回答>,
        "single_turn_metadata": <额外元数据>,
        "turns": [
            {
                "prompt": <历史消息列表>,
                "responses": [
                    {"completion": <助手回复>, "score": <质量分数>},
                    ...
                ]
            },
            ...
        ]
    }

    Args:
        session: TeachingSession 对象
        use_quality_score: 是否使用 session 的 quality_score 作为 responses 的分数
        extra_metadata: 额外的元数据字典，会与 session 元数据合并

    Returns:
        CollabLLM 嵌套格式的字典
    """
    if not session.dialogue:
        logger.warning(f"Session {session.student_id} has empty dialogue, skipping")
        return None

    # 1. 构建元数据
    metadata = {
        "source": "SID",
        "student_type": session.student_type,
        "scenario": session.scenario,
        "topic_id": session.topic_id,
        "student_id": session.student_id,
        "repeat_id": session.repeat_id,
        "quality_score": session.quality_score,
    }

    # 添加额外元数据
    if extra_metadata:
        metadata.update(extra_metadata)

    # 2. 生成 conv_id
    content_hash = get_content_hash([
        {"role": d.role, "content": d.content} for d in session.dialogue
    ])
    conv_id = (
        f"{session.topic_id}_{session.student_id}_{session.repeat_id}_{content_hash}"
    )

    # 3. 提取 single_turn_prompt 和 single_turn_completion
    # 通常：第一个学生消息作为 single_turn_prompt，对应的第一个教师回复作为 single_turn_completion
    single_turn_prompt = ""
    single_turn_completion = ""

    for i, turn in enumerate(session.dialogue):
        if normalize_role(turn.role) == "user" and not single_turn_prompt:
            single_turn_prompt = turn.content
        elif (
            normalize_role(turn.role) == "assistant"
            and single_turn_prompt
            and not single_turn_completion
        ):
            single_turn_completion = turn.content
            break

    # 如果没找到，使用 topic_text 或空字符串
    if not single_turn_prompt:
        single_turn_prompt = session.topic_text

    # 4. 构建多轮对话 turns
    turns = []
    history = []  # 累积的对话历史

    for i, turn in enumerate(session.dialogue):
        normalized_role = normalize_role(turn.role)

        # 当遇到教师（assistant）回复时，创建一个训练样本
        if normalized_role == "assistant":
            if history:  # 确保有历史对话
                # 优先级: 1. turn 自身的 score (单轮分)
                #        2. session 的 quality_score (全局分)
                #        3. 默认值 1.0
                if turn.score is not None:
                    score = turn.score
                elif use_quality_score:
                    score = session.quality_score
                else:
                    score = 1.0

                turns.append({
                    "prompt": list(history),
                    "responses": [
                        {"completion": turn.content, "score": score},
                    ],
                })

        # 将当前轮次加入历史（供后续轮次作为上下文）
        history.append({
            "role": normalized_role,
            "content": turn.content,
        })

    if not turns:
        logger.warning(f"Session {session.student_id} generated no turns, skipping")
        return None

    return {
        "conv_id": conv_id,
        "single_turn_prompt": single_turn_prompt,
        "single_turn_completion": single_turn_completion,
        "single_turn_metadata": metadata,
        "turns": turns,
    }


def convert_session_to_flat(
    session: TeachingSession,
    use_quality_score: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    将单个 TeachingSession 转换为 CollabLLM 扁平格式。

    扁平格式结构（每一行代表一个训练样本）：
    {
        "conv_id": <对话ID>,
        "prompt": <历史消息列表>,
        "completion": <助手回复>,
        "score": <质量分数>,
        "turn_id": <轮次编号>,
        "single_turn_prompt": <原始任务>,
        "single_turn_completion": <原始回答>,
        "single_turn_metadata": <元数据>,
        ...其他字段
    }

    Args:
        session: TeachingSession 对象
        use_quality_score: 是否使用 quality_score
        extra_metadata: 额外元数据

    Returns:
        扁平格式的字典列表
    """
    if not session.dialogue:
        return []

    # 构建元数据
    metadata = {
        "source": "SID",
        "student_type": session.student_type,
        "scenario": session.scenario,
        "topic_id": session.topic_id,
        "student_id": session.student_id,
        "repeat_id": session.repeat_id,
        "quality_score": session.quality_score,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    # 提取 single_turn
    single_turn_prompt = ""
    single_turn_completion = ""

    for i, turn in enumerate(session.dialogue):
        if normalize_role(turn.role) == "user" and not single_turn_prompt:
            single_turn_prompt = turn.content
        elif (
            normalize_role(turn.role) == "assistant"
            and single_turn_prompt
            and not single_turn_completion
        ):
            single_turn_completion = turn.content
            break

    if not single_turn_prompt:
        single_turn_prompt = session.topic_text

    # 转换为扁平格式
    flat_data = []
    history = []
    content_hash = get_content_hash([
        {"role": d.role, "content": d.content} for d in session.dialogue
    ])

    for turn_idx, turn in enumerate(session.dialogue):
        normalized_role = normalize_role(turn.role)

        if normalized_role == "assistant" and history:
            conv_id = f"{session.topic_id}_{session.student_id}_{session.repeat_id}_{content_hash}_t{turn_idx}"

            # 使用单轮 score 或全局 quality_score
            if turn.score is not None:
                score = turn.score
            elif use_quality_score:
                score = session.quality_score
            else:
                score = 1.0

            flat_data.append({
                "conv_id": conv_id,
                "prompt": list(history),
                "completion": turn.content,
                "score": score,
                "turn_id": len(history),  # 历史消息数 = 轮次号
                "single_turn_prompt": single_turn_prompt,
                "single_turn_completion": single_turn_completion,
                "single_turn_metadata": metadata,
            })

        history.append({
            "role": normalized_role,
            "content": turn.content,
        })

    return flat_data


# ============================================================================
# 批量转换函数
# ============================================================================


def convert_sessions_to_nested(
    sessions: List[TeachingSession],
    use_quality_score: bool = True,
    filter_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量转换多个 TeachingSession 为嵌套格式。

    Args:
        sessions: TeachingSession 列表
        use_quality_score: 是否使用质量分数
        filter_empty: 是否过滤转换失败的 session

    Returns:
        转换后的嵌套格式字典列表
    """
    converted = []
    failed = 0

    for session in sessions:
        try:
            result = convert_session_to_nested(
                session,
                use_quality_score=use_quality_score,
            )
            if result:
                converted.append(result)
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to convert session {session.student_id}: {e}")
            failed += 1

    logger.info(
        f"Batch conversion complete: {len(converted)} converted, {failed} failed"
    )
    return converted


def convert_sessions_to_flat(
    sessions: List[TeachingSession],
    use_quality_score: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量转换多个 TeachingSession 为扁平格式。

    Args:
        sessions: TeachingSession 列表
        use_quality_score: 是否使用质量分数

    Returns:
        转换后的扁平格式字典列表
    """
    converted = []
    failed = 0

    for session in sessions:
        try:
            result = convert_session_to_flat(
                session,
                use_quality_score=use_quality_score,
            )
            converted.extend(result)
        except Exception as e:
            logger.error(f"Failed to convert session {session.student_id}: {e}")
            failed += 1

    logger.info(
        f"Batch conversion complete: {len(converted)} samples generated, {failed} failed"
    )
    return converted


# ============================================================================
# 带聚合的转换（用于合并多个回复）
# ============================================================================


def convert_sessions_to_nested_with_aggregation(
    sessions: List[TeachingSession],
    use_quality_score: bool = True,
) -> List[Dict[str, Any]]:
    """
    将 TeachingSession 转换为嵌套格式，并对相同的 (conv_id, turn_id) 进行聚合。

    这样可以为同一轮对话保留多个不同的回复（responses 列表中有多条）。

    Args:
        sessions: TeachingSession 列表
        use_quality_score: 是否使用质量分数

    Returns:
        聚合后的嵌套格式字典列表
    """
    # 首先转换所有 session 为扁平格式
    flat_data = convert_sessions_to_flat(
        sessions,
        use_quality_score=use_quality_score,
    )

    # 按 (conv_id, turn_id) 聚合
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)

    for item in flat_data:
        key = (item["conv_id"], item["turn_id"])
        grouped[key].append(item)

    # 转换回嵌套格式
    nested_data = []

    for (conv_id, turn_id), items in grouped.items():
        # 使用第一条记录的元数据
        first = items[0]

        nested_data.append({
            "conv_id": conv_id,
            "single_turn_prompt": first["single_turn_prompt"],
            "single_turn_completion": first["single_turn_completion"],
            "single_turn_metadata": first["single_turn_metadata"],
            "turns": [
                {
                    "prompt": first["prompt"],
                    "responses": [
                        {
                            "completion": item["completion"],
                            "score": item["score"],
                        }
                        for item in items
                    ],
                }
            ],
        })

    logger.info(
        f"Aggregation complete: {len(nested_data)} unique (conv_id, turn_id) pairs, "
        f"with {len(flat_data)} total responses"
    )
    return nested_data
