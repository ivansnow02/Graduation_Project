#!/usr/bin/env python3
from pathlib import Path
import runpy


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[1]
        / "archive"
        / "legacy_engine"
        / "build_vanilla_dpo_dataset.py"
    )
    runpy.run_path(str(target), run_name="__main__")
