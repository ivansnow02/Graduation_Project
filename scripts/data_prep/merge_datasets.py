#!/usr/bin/env python3
"""
合并两个 DPO 数据集为一个混合数据集。
"""

import json
import argparse
from pathlib import Path


def merge_datasets(input_files, output_file):
    """
    合并多个 JSON 文件中的数据集。

    Args:
        input_files: 输入文件路径列表
        output_file: 输出文件路径
    """
    merged_data = []

    for input_file in input_files:
        print(f"Loading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  - Loaded {len(data)} samples")
            merged_data.extend(data)

    print(f"\nTotal merged samples: {len(merged_data)}")

    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存合并后的数据
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Total {len(merged_data)} samples saved.")

    # 打印统计信息
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(merged_data)}")

    # 检查样本结构
    if merged_data:
        sample = merged_data[0]
        print(f"\nSample structure:")
        print(f"  Keys: {list(sample.keys())}")
        if 'turns' in sample:
            avg_turns = sum(len(s.get('turns', [])) for s in merged_data) / len(merged_data)
            print(f"  Average turns per sample: {avg_turns:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple DPO datasets")
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Input JSON files to merge"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output merged JSON file"
    )

    args = parser.parse_args()
    merge_datasets(args.input_files, args.output_file)
