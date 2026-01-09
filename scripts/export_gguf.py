"""
Script to export an Unsloth-finetuned model to GGUF format.
Reference: https://github.com/unslothai/unsloth/wiki#saving-to-gguf

Usage:
  python3 collabllm/scripts/export_gguf.py --model_dir outputs/sid_unsloth_sft --quantization q4_k_m
"""

import argparse
import os
import sys

# Add workspace to path to ensure collabllm is findable if needed
sys.path.append("/workspace")
# ensure collabllm utilities on path
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":/workspace/collabllm"

os.environ["HF_ENDPOINT"] = (
    "https://hf-mirror.com"  # 使用 Hugging Face 镜像加速模型下载
)
from unsloth import FastLanguageModel


def export_to_gguf(model_dir, quantization_method):
    print(f"Loading model and tokenizer from {model_dir}...")

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print(f"Exporting to GGUF with quantization: {quantization_method}...")

    # Unsloth's built-in GGUF export
    # This will save the GGUF file in the model_dir
    model.save_pretrained_gguf(
        model_dir, tokenizer, quantization_method=quantization_method
    )

    print(f"Done! GGUF file should be in {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Unsloth model to GGUF")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/workspace/outputs/sid_unsloth_sft",
        help="Path to the finetuned model directory",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Quantization method (e.g., q4_k_m, q8_0, f16)",
    )

    args = parser.parse_args()

    export_to_gguf(args.model_dir, args.quantization)
