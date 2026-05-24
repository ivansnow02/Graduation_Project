import argparse
import os
import sys
from collabllm.datasets.multiturn import MultiturnDataset


def main():
    parser = argparse.ArgumentParser(description="将 Multiturn 数据集转换为 DPO 格式")
    parser.add_argument(
        "--input_file", type=str, required=True, help="输入 JSON/JSONL 文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="保存输出文件的目录"
    )
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="评估集拆分比例")

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：未找到输入文件 '{args.input_file}'。")
        sys.exit(1)

    print(f"Loading dataset from {args.input_file}...")
    ds = MultiturnDataset(args.input_file)

    print("正在转换为 DPO 格式...")
    dpo_ds = ds.to_dpo_dataset(eval_ratio=args.eval_ratio)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"保存到 {args.output_dir}...")
    # 如果需要可手动控制保存格式，否则可使用 save_to_disk / to_json
    # 这里手动保存以确保编码和格式（ensure_ascii=False）
    import json

    for split, dataset in dpo_ds.items():
        output_path = os.path.join(args.output_dir, f"{split}.json")
        print(f"Saving {split} split to {output_path} ({len(dataset)} examples)...")
        # Convert dataset split to a list of dicts
        data_list = [item for item in dataset]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
