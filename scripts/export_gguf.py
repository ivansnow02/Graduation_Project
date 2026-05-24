"""
将已合并的 16-bit 模型或 LoRA 适配器导出为 GGUF 格式。

对于已合并的 16-bit 模型：直接调用 llama.cpp 的转换脚本（无需 GPU）。
对于 LoRA 适配器：先使用 `merge_lora_to_base.py` 合并到基模型，再进行转换。

用法示例见下方注释。
"""

import argparse
import os
import subprocess
import sys
import json


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_CPP_DIR = os.path.join(PROJECT_ROOT, "llama.cpp")


def _clean_config_json(model_dir):
    """如果 config.json 中存在 `quantization_config` 字段，则将其移除（避免转换器报错）。"""
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
    """从 config.json 推断模型的数据类型（dtype），返回 'bf16' 或 'f16' 以供转换脚本使用。"""
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
    """使用 llama.cpp 的转换脚本将 HuggingFace 模型转换为 GGUF。"""
    converter = os.path.join(LLAMA_CPP_DIR, "unsloth_convert_hf_to_gguf.py")
    cmd = [
        sys.executable,
        converter,
        "--outfile",
        output_gguf,
        "--outtype",
        outtype,
        model_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def quantize_gguf(input_gguf, output_gguf, quant_type):
    """将 bf16 的 GGUF 文件量化为目标更小格式（如 q4_k_m）。"""
    quantizer = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
    cmd = [quantizer, input_gguf, output_gguf, quant_type]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_to_gguf(model_dir, quantization):
    _clean_config_json(model_dir)

    model_name = os.path.basename(os.path.normpath(model_dir))
    dtype = _get_model_dtype(model_dir)

    # 第一步：将 HF 模型转换为未量化的 GGUF
    base_gguf = os.path.join(model_dir, f"{model_name}.BF16.gguf")
    print(f"[1/2] Converting to {dtype} GGUF...")
    if os.path.exists(base_gguf):
        print(f"  Skipping — {base_gguf} already exists")
    else:
        convert_hf_to_gguf(model_dir, base_gguf, outtype=dtype)

    # 第二步：将 GGUF 量化为目标格式（如果需要）
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
        "--lora",
        action="store_true",
        help="Input is a LoRA adapter. Merge with base model first via merge_lora_to_base.py.",
    )
    args = parser.parse_args()

    if args.lora:
        merge_script = os.path.join(
            PROJECT_ROOT, "scripts", "train", "merge_lora_to_base.py"
        )
        merged_dir = args.model_dir.rstrip("/") + "_merged_16bit"
        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            print("Merging LoRA adapter first...")
            subprocess.run(
                [
                    sys.executable,
                    merge_script,
                    "--adapter_dir",
                    args.model_dir,
                    "--output_dir",
                    merged_dir,
                ],
                check=True,
            )
        else:
            print(f"Using existing merged model: {merged_dir}")
        model_dir = merged_dir
    else:
        model_dir = args.model_dir

    export_to_gguf(model_dir, args.quantization)
