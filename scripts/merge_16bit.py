from unsloth import FastLanguageModel
import torch

# ========== 请确认以下三个路径 ==========
BASE_MODEL = "Qwen/Qwen3-14B"             # 最原始的 16-bit 官方底座
SFT_LORA = "outputs/sid_unsloth_sft"      # 你的 SFT LoRA 路径
DPO_LORA = "outputs/dpo_model_1k_3can_opt" # 你的 DPO LoRA 路径
OUTPUT_DIR = "/root/autodl-fs/qwen14b_socratic_16bit_final"
# ========================================

print("🚀 步骤 1: 正在加载 16-bit 原始底座 + SFT LoRA...")
# 🚨 关键改变：关闭 4-bit，显式使用 bfloat16 加载！
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=4096,
    dtype=torch.bfloat16,  # RTX 5090 原生支持的最佳精度
    load_in_4bit=False,    # 绝对不能设为 True！
)

# 挂载 SFT LoRA
model.load_adapter(SFT_LORA)
print("🧩 正在将 SFT 策略物理融合进底座...")
# 注意：这一步是在显存里做矩阵加法
model = model.merge_and_unload()

print("🧠 步骤 2: 正在挂载 DPO (苏格拉底) LoRA...")
# 继续挂载 DPO LoRA
model.load_adapter(DPO_LORA)
print("🧩 正在将 DPO 策略物理融合进底座...")
model = model.merge_and_unload()


