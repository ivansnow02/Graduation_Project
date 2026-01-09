"""Analyze annotation field distributions using the TeachingSession dataclass."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


from collabllm.datasets.types import TeachingSession


def split_strategies(value: str) -> Iterable[str]:
    if not value:
        return []
    separators = "，,;；"  # common separators
    parts = [value]
    for sep in separators:
        new_parts: List[str] = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    return (part.strip() for part in parts if part.strip())


def analyze(base_dir: Path) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Path does not exist: {base_dir}")

    counters = {
        "cognitive_level": Counter(),
        "teacher_intent": Counter(),
        "teaching_strategy": Counter(),
    }

    total_sessions = 0
    processed_annotations = 0

    for path in sorted(base_dir.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    session = TeachingSession.from_dict(obj)
                except Exception as exc:  # pragma: no cover
                    print(f"warning: failed to parse {path}: {exc}")
                    continue

                total_sessions += 1
                for ann in session.annotations:
                    processed_annotations += 1
                    counters["cognitive_level"][ann.cognitive_level.strip()] += 1
                    counters["teacher_intent"][ann.teacher_intent.strip()] += 1
                    for strategy in split_strategies(ann.teaching_strategy):
                        counters["teaching_strategy"][strategy] += 1

    if total_sessions == 0:
        print("No sessions found under", base_dir)
        return

    print("\nAnalysis complete")
    print("Total sessions:", total_sessions)
    print("Total annotation turns:", processed_annotations)

    for field in ["cognitive_level", "teacher_intent", "teaching_strategy"]:
        print(f"\nTop 50 values for {field}:")
        for value, count in counters[field].most_common(50):
            print(f"  {count:6}  {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze annotation field distribution from data/annotated."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data") / "annotated",
        help="Root directory containing annotated JSONL files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.path)


if __name__ == "__main__":
    main()
