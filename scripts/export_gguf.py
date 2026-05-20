"""
Export a merged 16-bit or LoRA model to GGUF format.

For already-merged 16-bit models: directly calls llama.cpp converter (no GPU needed).
For LoRA adapters: merges with merge_lora_to_base.py first, then converts.

Usage:
    # From already-merged 16-bit model (fast, low memory):
    python scripts/export_gguf.py --model_dir outputs/dpo500softmargin_merged_16bit --quantization q4_k_m

    # From LoRA adapter (needs base model download + merge first):
    python scripts/export_gguf.py --model_dir outputs/dpo500softmargin --quantization q4_k_m --lora
"""

import argparse
import os
import subprocess
import sys
import json


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_CPP_DIR = os.path.join(PROJECT_ROOT, "llama.cpp")


def _clean_config_json(model_dir):
    """Remove quantization_config from config.json if present."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("  Stripped quantization_config from config.json")


def _get_model_dtype(model_dir):
    """Infer model dtype from config.json. Returns 'bf16' or 'f16' for the converter."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    dtype = cfg.get("torch_dtype", "bfloat16")
    if dtype in (None, "bfloat16"):
        return "bf16"
    if dtype == "float16":
        return "f16"
    return "bf16"


def convert_hf_to_gguf(model_dir, output_gguf, outtype="bf16"):
    """Convert HuggingFace model to GGUF using llama.cpp converter."""
    converter = os.path.join(LLAMA_CPP_DIR, "unsloth_convert_hf_to_gguf.py")
    cmd = [
        sys.executable, converter,
        "--outfile", output_gguf,
        "--outtype", outtype,
        model_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def quantize_gguf(input_gguf, output_gguf, quant_type):
    """Quantize a bf16 GGUF file to a smaller format."""
    quantizer = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
    cmd = [quantizer, input_gguf, output_gguf, quant_type]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_to_gguf(model_dir, quantization):
    _clean_config_json(model_dir)

    model_name = os.path.basename(os.path.normpath(model_dir))
    dtype = _get_model_dtype(model_dir)

    # Step 1: Convert HF model to unquantized GGUF
    base_gguf = os.path.join(model_dir, f"{model_name}.BF16.gguf")
    print(f"[1/2] Converting to {dtype} GGUF...")
    if os.path.exists(base_gguf):
        print(f"  Skipping — {base_gguf} already exists")
    else:
        convert_hf_to_gguf(model_dir, base_gguf, outtype=dtype)

    # Step 2: Quantize to target format
    if quantization in ("f16", "bf16", "f32"):
        print(f"[2/2] No quantization needed — output is {base_gguf}")
        return

    quant_gguf = os.path.join(model_dir, f"{model_name}-{quantization.upper()}.gguf")
    print(f"[2/2] Quantizing to {quantization}...")
    if os.path.exists(quant_gguf):
        print(f"  Skipping — {quant_gguf} already exists")
    else:
        quantize_gguf(base_gguf, quant_gguf, quantization)

    print(f"\nDone! GGUF: {quant_gguf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export model to GGUF")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="q4_k_m")
    parser.add_argument(
        "--lora", action="store_true",
        help="Input is a LoRA adapter. Merge with base model first via merge_lora_to_base.py.",
    )
    args = parser.parse_args()

    if args.lora:
        merge_script = os.path.join(PROJECT_ROOT, "scripts", "train", "merge_lora_to_base.py")
        merged_dir = args.model_dir.rstrip("/") + "_merged_16bit"
        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            print("Merging LoRA adapter first...")
            subprocess.run(
                [sys.executable, merge_script,
                 "--adapter_dir", args.model_dir,
                 "--output_dir", merged_dir],
                check=True,
            )
        else:
            print(f"Using existing merged model: {merged_dir}")
        model_dir = merged_dir
    else:
        model_dir = args.model_dir

    export_to_gguf(model_dir, args.quantization)
