# SID Learning Platform: Socratic Interdisciplinary Dialogue

基于多轮对话生成、苏格拉底式教学质量标注、奖励感知数据构造以及 DPO/SFT 微调，训练和评估面向跨学科启发式教学的协作型大语言模型。

[English](README.md)

<div align="left">

[![](https://img.shields.io/badge/Framework-CollabLLM-purple?style=plastic&logo=Google%20Chrome)](https://github.com/microsoft/collabllm)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.00640)
[![Python](https://img.shields.io/badge/Python-3.11--3.12-blue.svg)](https://www.python.org/)
[![Vue](https://img.shields.io/badge/Web_Demo-Vue_3-42b883.svg)](https://vuejs.org/)

</div>

## 项目简介

本项目围绕 **SID（Socratic Interdisciplinary Dialogue，苏格拉底式跨学科对话）** 构建完整实验管线：从跨学科主题出发，生成教师与学生之间的多轮教学对话；使用 LLM 对教学质量进行结构化标注；将标注结果转化为偏好数据；最后通过 SFT 或 DPO 微调模型，使模型更倾向于提出启发式问题、引导知识迁移、纠正认知偏差，并保持多轮教学连贯性。

项目扩展自 **CollabLLM** 思路，重点关注“模型是否能主动协作教学”，而不仅是被动回答问题。

## 核心能力

- **跨学科主题驱动**：面向两个或多个学科之间的概念关联生成教学任务。
- **多轮苏格拉底式对话**：通过教师提问、学生模拟回答、教师继续引导的方式构造完整教学过程。
- **教学质量标注**：使用 LLM 或批量 API 对对话进行结构化评分与错误分析。
- **多维客观评测**：聚合策略密度、策略多样性、跨学科迁移、Bloom 认知进阶、结构完整性等指标。
- **偏好数据构造**：从多个候选回复中选择更优教学行为，生成 DPO 所需的 chosen/rejected 数据。
- **模型训练与导出**：支持 Unsloth 加速的 SFT/DPO、LoRA 合并、GGUF 导出，以及 Hugging Face / ModelScope 发布。
- **Web 演示端**：提供 Vue 3 + Vite 前端，用于交互式教学对话、配置模型、查看/编辑评测标注。

## 仓库结构

```text
.
├── collabllm/                         # 核心 Python 包
│   ├── synthetic.py                   # 多轮合成数据生成入口
│   ├── simulation.py                  # 对话模拟
│   ├── reward.py                      # 多轮奖励与偏好计算
│   ├── metric.py                      # 通用指标封装
│   ├── metrics/teaching_quality.py    # SID 教学质量指标
│   ├── datasets/                      # 数据集加载、清洗与转换
│   ├── modules/                       # LLM 协作者与学生模拟器
│   └── prompts/                       # 教师、学生、抽取等提示词模板
│
├── scripts/
│   ├── build_dpo_dataset.sh           # build/train/eval/all 一体化实验脚本
│   ├── run_benchmark.sh               # batch 标注与客观评测流程
│   ├── engine/                        # 生成与推理脚本
│   ├── benchmark/                     # 标注与客观评测
│   ├── converter/                     # 标注数据转 SFT/DPO 数据
│   ├── data_prep/                     # 外部数据清洗、重写、混合
│   └── train/                         # SFT、DPO、LoRA 合并脚本
│
├── web_demo/frontend/                 # Vue 3 + Vite 前端演示
├── pyproject.toml                     # Python 包配置
├── requirements.txt                   # 已冻结依赖快照
└── uv.lock                            # uv 锁文件
```

> 注意：当前仓库不强制包含 `data/` 和 `outputs/`。这些目录通常在生成数据、标注、训练或评测后创建。

## 环境要求

- Python `>=3.11, <3.13`
- 推荐使用 `uv` 管理 Python 环境
- CUDA GPU 环境用于 Unsloth、vLLM、DPO/SFT 训练
- Web Demo 推荐使用 Bun；也可以使用 npm/pnpm/yarn
- 如果使用在线模型或批量标注，需要配置对应 API Key

## 安装

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

如果需要完全复现当前冻结环境，可使用：

```bash
uv pip install -r requirements.txt
```

训练相关依赖体积较大，并且与 CUDA、PyTorch、Unsloth、vLLM 版本强相关。若安装失败，优先检查本机 CUDA/PyTorch 兼容性。

## 环境变量

项目脚本会读取 `.env`：

```bash
cp .env.example .env  # 如果仓库中存在该模板
```

常见配置包括：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
HF_TOKEN=your_huggingface_token
MODELSCOPE_TOKEN=your_modelscope_token
```

本地 OpenAI-compatible 服务也可以通过脚本参数传入，例如 `BASE_URL=http://localhost:8000/v1`。

## 快速开始

一体化脚本支持 `build`、`train`、`eval`、`all`：

```bash
./scripts/build_dpo_dataset.sh build dpo_500_sid
./scripts/build_dpo_dataset.sh train dpo_500_sid
./scripts/build_dpo_dataset.sh eval dpo_500_sid qwen-flash
```

常用环境变量覆盖：

```bash
BASE_MODEL=/path/to/base-model \
BASE_URL=http://localhost:8000/v1 \
REWARD_MODEL=openai/qwen-flash \
MODEL_NAME=outputs/sid_qwen14b_sft_2500 \
TRAIN_SIZE=500 \
NUM_CANDIDATES=3 \
EPOCHS=3 \
./scripts/build_dpo_dataset.sh all dpo_500_sid qwen-flash
```

默认输出：

```text
outputs/<experiment>/interdisciplinary_multiturn.json
outputs/<experiment>_opt/
outputs/logs/<experiment>_*.log
```

## 手动生成多轮对话数据

```bash
uv run scripts/engine/build_dataset.py \
  --dataset_name interdisciplinary \
  --metric_names teaching_quality \
  --metric_weights 1.0 \
  --num_candidate_responses 3 \
  --train_size 500 \
  --output_dir outputs/dpo_500_sid \
  --user_prompt_file collabllm/prompts/student_simulator.txt \
  --user_generation_kwargs '{"model":"Qwen/Qwen3-14B","base_url":"http://localhost:8000/v1","api_key":"not-needed","temperature":1.0,"max_tokens":2048}' \
  --assistant_generation_kwargs '{"model":"Qwen/Qwen3-14B","base_url":"http://localhost:8000/v1","api_key":"not-needed","temperature":1.0,"max_tokens":2048}' \
  --reward_generation_kwargs '{"model":"qwen-flash","temperature":0}' \
  --resume
```

## 标注与客观评测

单机逐条标注：

```bash
ANNOTATION_INPUT_DIR=data/dialog/dpo_500_sid \
ANNOTATION_OUTPUT_DIR=data/final/dpo_500_sid \
uv run scripts/benchmark/annotation.py
```

Batch 标注流程：

```bash
./scripts/run_benchmark.sh prepare dpo_500_sid qwen-flash
# 上传 data/batch/dpo_500_sid/batch.jsonl，等待平台处理
# 下载 batch 输出到 data/batchoutput/dpo_500_sid/
./scripts/run_benchmark.sh eval dpo_500_sid
```

客观指标汇总：

```bash
uv run scripts/benchmark/objective_eval.py \
  data/final/dpo_500_sid/*.jsonl \
  --summary-file data/metrics/dpo_500_sid.json \
  --output-dir data/metrics/dpo_500_sid_per_file_metrics
```

## 训练

SFT：

```bash
uv run scripts/train/sft_unsloth.py \
  --dataset_repo data/sft/train.json \
  --output_dir outputs/sid_sft \
  --model_name Qwen/Qwen3-14B
```

DPO：

```bash
uv run scripts/train/offline_dpo_unsloth.py \
  --dataset_repo outputs/dpo_500_sid/interdisciplinary_multiturn.json \
  --output_dir outputs/sid_dpo \
  --model_name outputs/sid_sft \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --max_seq_length 4096 \
  --use_swanlab
```

## 模型合并与 GGUF 导出

合并 LoRA：

```bash
uv run scripts/train/merge_lora_to_base.py \
  --base_model /path/to/base-model \
  --lora_dir outputs/sid_dpo \
  --output_dir outputs/sid_dpo_merged
```

导出 GGUF：

```bash
uv run scripts/export_gguf.py \
  --model_dir outputs/sid_dpo_merged \
  --quantization q4_k_m
```

## Web Demo

前端位于 `web_demo/frontend`，用于本地交互式测试、模型配置、对话记录和教学质量标注编辑。

```bash
cd web_demo/frontend
bun install
bun run dev
```

构建生产版本：

```bash
bun run build
```

如果本机没有 Bun，可以改用 npm：

```bash
npm install
npm run dev
```

前端主要能力：

- 配置 OpenAI-compatible 模型服务
- 进行教师/学生式对话交互
- 展示评测面板
- 编辑结构化标注
- 导出当前会话与评测 JSONL

## 数据格式

跨学科主题示例：

```json
[
  {
    "topic": "How does photosynthesis relate to climate change?",
    "discipline_a": "Biology",
    "discipline_b": "Environmental Science"
  }
]
```

生成后的多轮数据通常包含：

- 对话上下文
- 教师候选回复
- 学生模拟回复
- 教学质量评分
- chosen / rejected 偏好对
- 用于训练和评测的元数据

## 教学质量指标

项目中的 SID 评测主要关注以下维度：

- **StrategyDensity**：教学策略使用密度
- **StrategyVariety**：教学策略多样性
- **IKT**：跨学科知识迁移能力
- **BloomProgression**：Bloom 认知层级推进
- **StructureCompleteness**：对话结构完整性
- **L3GuidanceRate**：有效提示与引导比例
- **CognitiveCorrectionRate**：认知错误纠正质量
- **ResponseRelevance**：回复相关性与任务贴合度

## 发布模型

当前 checkout 没有内置模型上传脚本。发布前建议先确认要上传的是 LoRA adapter、合并后的完整模型，还是 GGUF 文件，并排除不需要的 `checkpoint-*` 中间目录。Hugging Face 可使用 `huggingface-cli upload` 或 `huggingface_hub` SDK；ModelScope 可使用官方 SDK。

## 常见问题

**1. 为什么生成或训练脚本需要本地模型服务？**  
多轮数据生成通常需要教师模型、学生模拟模型和奖励/标注模型。脚本支持通过 OpenAI-compatible `base_url` 连接本地 vLLM、远程 API 或其他兼容服务。

**2. `data/` 目录不存在怎么办？**  
这是正常情况。`data/`、`outputs/`、`data/batch/`、`data/final/` 等目录会在数据准备、标注、评测或训练时创建。

**3. `requirements.txt` 和 `pyproject.toml` 应该用哪个？**  
开发安装优先使用 `uv pip install -e .`；如果需要复现某次完整环境，再参考 `requirements.txt`。训练依赖建议根据 GPU/CUDA 环境调整。

**4. Web Demo 的数据保存在哪里？**  
前端会把配置、会话和评测状态保存在浏览器本地存储中。换浏览器或换机器后不会自动同步。

## 引用

如果你使用本项目，请引用 SID benchmark 论文：

```bibtex
@article{jiang2025sid,
    title={SID: Benchmarking Guided Instruction Capabilities in STEM Education with a Socratic Interdisciplinary Dialogues Dataset},
    author={Jiang, Mei and Yue, Houping and Li, Bingdong and Hao, Hao and Qian, Ying and Jiang, Bo and Zhou, Aimin},
    journal={arXiv preprint arXiv:2508.04563},
    year={2025}
}
```

本项目也基于 CollabLLM 思路：

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and Gavin Li and Yao Dou and Weixin Cai and James Zou and Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```
