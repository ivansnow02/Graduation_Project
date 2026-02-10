import argparse
import os
import sys
from collabllm.datasets.multiturn import MultiturnDataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert Multiturn Dataset to DPO format"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input JSON/JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.1, help="Evaluation split ratio"
    )

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Loading dataset from {args.input_file}...")
    ds = MultiturnDataset(args.input_file)

    print("Converting to DPO format...")
    dpo_ds = ds.to_dpo_dataset(eval_ratio=args.eval_ratio)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving to {args.output_dir}...")
    # Save manually to control format if needed, or use save_to_disk / to_json
    # Save manually to control format (ensure_ascii=False)
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
