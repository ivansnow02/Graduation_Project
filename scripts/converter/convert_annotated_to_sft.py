#!/usr/bin/env python3
"""
将清洗后的 TeachingSession 数据转换为 CollabLLM 标准格式的脚本。

支持多种输入和输出格式：
- 输入：JSON/JSONL 文件（包含 TeachingSession 数据）或已清洗的对象列表
- 输出：CollabLLM 嵌套格式（用于 MultiturnDataset）或扁平格式

用法示例：
- 转换为嵌套格式（推荐）
- 转换为扁平格式
- 进行聚合（多个回复合并）
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collabllm.datasets.types import TeachingSession
from collabllm.datasets.converter import (
    convert_sessions_to_nested,
    convert_sessions_to_flat,
    convert_sessions_to_nested_with_aggregation,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 数据加载


def load_teaching_sessions(path: str) -> List[TeachingSession]:
    """
    从 JSON/JSONL 文件加载 TeachingSession 对象。

        Args:
            path: 文件路径（.json 或 .jsonl）或目录（递归查找所有 .json/.jsonl）。

        Returns:
            TeachingSession 列表。
    """
    sessions = []

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith((".json", ".jsonl")):
                    files.append(os.path.join(root, filename))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    for filepath in sorted(files):
        logger.info(f"Loading: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".json"):
                    data = json.load(f)
                    # 处理单个对象或列表
                    if isinstance(data, dict):
                        data = [data]
                else:  # .jsonl
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

            # 转换为 TeachingSession 对象
            for raw_dict in data:
                try:
                    session = TeachingSession.from_dict(raw_dict)
                    sessions.append(session)
                except Exception as e:
                    logger.error(f"Failed to parse TeachingSession: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue

    logger.info(f"Loaded {len(sessions)} TeachingSession objects")
    return sessions


def save_json(data: Any, path: str) -> None:
    """保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {path}")


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """保存为 JSONL 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} items to {path}")


# 主程序


def main():
    parser = argparse.ArgumentParser(
        description="Convert TeachingSession data to CollabLLM standard format"
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Single JSON or JSONL file containing TeachingSession data",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing JSON/JSONL files (recursive search)",
    )

    # 输出参数
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output file path (.json or .jsonl)",
    )

    # 转换格式
    parser.add_argument(
        "--format",
        type=str,
        choices=["nested", "flat", "nested_agg"],
        default="nested",
        help=(
            "Output format: "
            "nested (recommended, for MultiturnDataset), "
            "flat (flattened list), "
            "nested_agg (nested with response aggregation)"
        ),
    )

    # 其他选项
    parser.add_argument(
        "--use_quality_score",
        action="store_true",
        default=True,
        help="Use TeachingSession.quality_score for response scores (default: True)",
    )
    parser.add_argument(
        "--no_quality_score",
        action="store_true",
        help="Override: set all response scores to 1.0",
    )

    args = parser.parse_args()

    # 确定输入路径
    input_path = args.input_file or args.input_dir

    # 加载数据
    logger.info(f"Loading TeachingSession from {input_path}...")
    sessions = load_teaching_sessions(input_path)

    if not sessions:
        logger.error("No sessions loaded, exiting")
        return

    # 决定是否使用质量分数
    use_quality_score = args.use_quality_score and not args.no_quality_score

    # 转换
    logger.info(f"Converting {len(sessions)} sessions to {args.format} format...")

    if args.format == "nested":
        converted = convert_sessions_to_nested(
            sessions,
            use_quality_score=use_quality_score,
        )
        save_json(converted, args.output_path)

    elif args.format == "flat":
        converted = convert_sessions_to_flat(
            sessions,
            use_quality_score=use_quality_score,
        )
        # 保存为 JSONL
        if args.output_path.endswith(".jsonl"):
            save_jsonl(converted, args.output_path)
        else:
            # 如果是 .json，保存为数组
            save_json(converted, args.output_path)

    elif args.format == "nested_agg":
        converted = convert_sessions_to_nested_with_aggregation(
            sessions,
            use_quality_score=use_quality_score,
        )
        save_json(converted, args.output_path)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
