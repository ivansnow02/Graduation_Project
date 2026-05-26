"""
collabllm.datasets
~~~~~~~~~~~~~~~~~~
数据集相关模块的统一导出入口。
"""

from .single_turn import SingleTurnDataset
from .multiturn import MultiturnDataset
from .types import TeachingSession, DialogueTurn, Annotation
from .converter import (
    convert_session_to_nested,
    convert_session_to_flat,
    convert_sessions_to_nested,
    convert_sessions_to_flat,
    convert_sessions_to_nested_with_aggregation,
)
from .cleaner import DataCleaner

__all__ = [
    "SingleTurnDataset",
    "MultiturnDataset",
    "TeachingSession",
    "DialogueTurn",
    "Annotation",
    "convert_session_to_nested",
    "convert_session_to_flat",
    "convert_sessions_to_nested",
    "convert_sessions_to_flat",
    "convert_sessions_to_nested_with_aggregation",
    "DataCleaner",
]
