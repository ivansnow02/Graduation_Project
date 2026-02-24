import torch
from unsloth import FastLanguageModel
import os

# 配置参数（与你 DPO 脚本保持一致）
MODEL_NAME = "outputs/qwen14b_warmup_merged_4bit"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
LORA_RANK = 64
BATCH_SIZE = 2  # 你想测试的 Batch Size

def run_test():
    print(f"🚀 正在测试模型: {MODEL_NAME}")
    print(f"📏 测试参数: Batch Size={BATCH_SIZE}, Max Length={MAX_SEQ_LENGTH}, Rank={LORA_RANK}")

    # 1. 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. 挂载全量 LoRA (模拟 DPO 的 all-linear 模式)
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_RANK * 2,
        lora_dropout = 0,
        bias = "none",
    )

    # 3. 构造极限长度的假数据 (全 1 填充)
    # DPO 实际上是对 (Prompt + Chosen) 和 (Prompt + Rejected) 同时计算
    # 所以显存占用会比普通 SFT 略高，这里我们模拟一个大 Batch
    input_ids = torch.ones((BATCH_SIZE, MAX_SEQ_LENGTH), dtype=torch.long).cuda()

    print("🔥 正在执行前向传播压力测试...")
    print("🔥 正在执行前向传播压力测试...")
    try:
        # 强制使用 bfloat16 以匹配 5090 的原生支持和模型精度
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids)
            # DPO 实际上会处理两倍长度（Chosen + Rejected），这里模拟一下
            loss = outputs.logits.mean()
            loss.backward()

        used_vram = torch.cuda.max_memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n✅ 测试成功！")
        print(f"📊 显存峰值占用: {used_vram:.2f} GB")
        print(f"🖥️  GPU 总显存: {total_vram:.2f} GB")

        if used_vram > total_vram * 0.95:
            print("⚠️  警告：显存占用极高（>95%），建议将 Batch Size 降至 2 并开启更大的 Gradient Accumulation。")
        else:
            print("🟢 显存充足，可以开始 DPO 训练！")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ 测试失败: 发生 OOM (显存溢出)！")
            print("📉 建议：请将 Batch Size 改为 2。")
        else:
            print(f"❌ 发生其他错误: {e}")

if __name__ == "__main__":
    run_test()
