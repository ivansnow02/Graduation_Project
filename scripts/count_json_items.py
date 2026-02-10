#!/usr/bin/env python3
"""Count JSON items in a file.

Supports:
- A single JSON array file
- A single JSON object file
- JSONL/NDJSON (one JSON object per line)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _count_from_full_parse(text: str) -> int | None:
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return 1
    return 1


def _count_from_jsonl(text: str) -> int:
    count = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
        count += 1
    return count


def count_json_items(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    count = _count_from_full_parse(text)
    if count is not None:
        return count
    return _count_from_jsonl(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count JSON items in a file.")
    parser.add_argument("path", type=Path, help="Path to a JSON/JSONL file")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File not found: {args.path}")

    total = count_json_items(args.path)
    print(total)


if __name__ == "__main__":
    main()
