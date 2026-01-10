# TeachingSession 到 CollabLLM 数据格式转换指南

## 概述

本指南说明如何将 `TeachingSession` 对象转换为 CollabLLM 标准的多轮数据格式，以用于训练协作型语言模型。

## 数据格式对比

### TeachingSession 格式（原始数据）

```python
{
    "student_id": "stu_001",
    "student_type": "高中生",
    "scenario": "一对一辅导",
    "topic_id": "math_001",
    "topic_text": "如何理解二次函数？",
    "repeat_id": "v1",
    "dialogue": [
        {"role": "学生", "content": "..."},
        {"role": "教师", "content": "..."},
        ...
    ],
    "annotations": [
        {"speaker": "学生", "utterance": "...", ...},
        ...
    ],
    "quality_score": 0.85
}
```

### CollabLLM 嵌套格式（推荐）

```python
{
    "conv_id": "math_001_stu_001_v1_a1b2c3d4",  # 唯一标识
    "single_turn_prompt": "如何理解二次函数？",  # 原始问题
    "single_turn_completion": "",  # 原始回答（可选）
    "single_turn_metadata": {  # 元数据
        "source": "SID",
        "student_type": "高中生",
        "scenario": "一对一辅导",
        "quality_score": 0.85,
        ...
    },
    "turns": [  # 多轮对话
        {
            "prompt": [  # 当前轮的历史对话
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ],
            "responses": [  # 可选的多个回复
                {"completion": "...", "score": 0.85},
                {"completion": "...", "score": 0.75},  # 可能的备选回复
            ]
        },
        ...
    ]
}
```

### CollabLLM 扁平格式（备选）

```python
{
    "conv_id": "math_001_stu_001_v1_...",
    "prompt": [{"role": "user", "content": "..."}, ...],
    "completion": "...",
    "score": 0.85,
    "turn_id": 2,  # 历史消息数
    "single_turn_prompt": "...",
    "single_turn_completion": "...",
    "single_turn_metadata": {...},
}
```

## 转换方法

### 方法 1: 命令行脚本（推荐用于批量处理）

```bash
# 转换为嵌套格式（推荐）
python scripts/converter/convert_annotated_to_sft.py \
    --input_dir data/cleaned \
    --output_path data/collabllm/nested.json \
    --format nested

# 转换为扁平格式
python scripts/converter/convert_annotated_to_sft.py \
    --input_file data/cleaned.jsonl \
    --output_path data/collabllm/flat.jsonl \
    --format flat

# 进行聚合（多个回复合并）
python scripts/converter/convert_annotated_to_sft.py \
    --input_dir data/cleaned \
    --output_path data/collabllm/aggregated.json \
    --format nested_agg
```

**命令行参数说明：**

- `--input_file`: 单个 JSON/JSONL 文件路径
- `--input_dir`: 包含多个 JSON/JSONL 文件的目录（递归扫描）
- `--output_path`: 输出文件路径（.json 或 .jsonl）
- `--format`: 输出格式
  - `nested`: 嵌套格式（**推荐**，用于 MultiturnDataset）
  - `flat`: 扁平格式（逐样本列表）
  - `nested_agg`: 嵌套格式+多回复聚合
- `--use_quality_score`: 使用 quality_score（默认 True）
- `--no_quality_score`: 覆盖，所有 score 设为 1.0

### 方法 2: Python API（用于编程集成）

#### 单个 session 转换

```python
from collabllm.datasets import (
    TeachingSession,
    convert_session_to_nested,
    convert_session_to_flat,
)

# 加载数据
raw_dict = {...}  # JSON 数据
session = TeachingSession.from_dict(raw_dict)

# 转换为嵌套格式
nested = convert_session_to_nested(session, use_quality_score=True)

# 转换为扁平格式
flat = convert_session_to_flat(session, use_quality_score=True)
```

#### 批量 session 转换

```python
from collabllm.datasets import (
    convert_sessions_to_nested,
    convert_sessions_to_flat,
    convert_sessions_to_nested_with_aggregation,
)

# 假设已有 sessions 列表
sessions = [...]

# 批量转换为嵌套格式
nested_data = convert_sessions_to_nested(sessions, use_quality_score=True)

# 批量转换为扁平格式
flat_data = convert_sessions_to_flat(sessions, use_quality_score=True)

# 进行多回复聚合
aggregated_data = convert_sessions_to_nested_with_aggregation(sessions)
```

#### 从文件加载并转换

```python
import json
from collabllm.datasets import TeachingSession, convert_sessions_to_nested

# 加载文件
with open("data/cleaned.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# 转换为 TeachingSession
sessions = [TeachingSession.from_dict(item) for item in raw_data]

# 转换为 CollabLLM 格式
converted = convert_sessions_to_nested(sessions)

# 保存
with open("data/collabllm_data.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)
```

## 与 MultiturnDataset 的集成

转换后的嵌套格式可以直接被 `MultiturnDataset` 加载：

```python
from collabllm.datasets import MultiturnDataset

# 加载转换后的数据
dataset = MultiturnDataset("data/collabllm_data.json")

# 查看数据集大小
print(f"Dataset size: {len(dataset)}")

# 转换为 SFT 格式（用于有监督微调）
sft_ds = dataset.to_sft_dataset(eval_ratio=0.1)
print(f"Train: {len(sft_ds['train'])}, Eval: {len(sft_ds['eval'])}")

# 转换为 DPO 格式（用于直接偏好优化）
dpo_ds = dataset.to_dpo_dataset(minimum_gap=0.1)

# 转换为输入数据集（不含回复，用于推理）
input_ds = dataset.to_inputs_dataset()
```

## 关键概念说明

### conv_id（对话ID）

- **组成**：`{topic_id}_{student_id}_{repeat_id}_{content_hash}`
- **作用**：唯一标识一个对话及其历史上下文
- **哈希部分**：基于对话内容计算，保证相同对话生成相同 ID

### turn_id（轮次ID）

- **定义**：当前 `prompt` 中的消息总数
- **示例**：如果 `prompt` 有 2 条消息，则 `turn_id = 2`
- **用途**：在 SFT/DPO 转换时，用于选择最优样本或配对

### score（质量分数）

- **双层评分体系**：
  - **单轮分数 (Turn Score)**：存储在 `DialogueTurn.score` 中，表示该特定回复的即时质量。在 CollabLLM 格式中，它被直接映射到 `responses` 列表下的 `score` 字段。
  - **会话分数 (Session Quality Score)**：存储在 `TeachingSession.quality_score` 中，是对整个对话过程的综合统计评价（如 SID 论文中的公式）。
- **转换逻辑**：
  1. 优先使用 `DialogueTurn` 中的单轮 `score`。
  2. 如果单轮分数缺失，则回退使用 `TeachingSession.quality_score`。
  3. 如果全局分数也缺失，默认设为 1.0。
- **差异说明**：单轮分数更适合强化学习（RL）的即时奖励，而会话分数更适合 SFT 阶段的整体样本筛选。

### 多回复聚合

当多个 TeachingSession 拥有相同的学生问题但教师回复不同时，可以使用聚合模式：

```python
aggregated = convert_sessions_to_nested_with_aggregation(sessions)

# 结果中，单个 turn 的 responses 包含多个备选项：
{
    "prompt": [...],
    "responses": [
        {"completion": "回复版本1", "score": 0.9},
        {"completion": "回复版本2", "score": 0.7},
        {"completion": "回复版本3", "score": 0.85},
    ]
}
```

这种形式特别适合 DPO 训练，可以直接从多个备选中选择 chosen 和 rejected。

## 最佳实践

### 1. 数据流程建议

```
原始 SID 数据
    ↓
[清洗] DataCleaner
    ↓
TeachingSession 列表（带 quality_score）
    ↓
[转换] convert_sessions_to_nested
    ↓
CollabLLM 嵌套格式 JSON
    ↓
[加载] MultiturnDataset
    ↓
[转换] to_sft_dataset / to_dpo_dataset
    ↓
训练数据
```

### 2. 选择转换格式

| 格式       | 用途                                | 优点                          | 缺点              |
| ---------- | ----------------------------------- | ----------------------------- | ----------------- |
| nested     | MultiturnDataset 加载、SFT/DPO 训练 | 保留结构，易处理多个 response | 文件较大          |
| flat       | 直接扁平化处理、简单数据分析        | 简单直接、占用空间少          | 丢失结构信息      |
| nested_agg | 多个 response 聚合、DPO 训练准备    | 自动合并备选 response         | 需要相同的 prompt |

### 3. 质量分数的使用

```python
# 创建高质量数据集（只保留分数 ≥ 0.7 的样本）
sft_ds = dataset.to_sft_dataset(
    lower_bound_metric="single_turn_metadata.quality_score",
    lower_bound=0.7
)

# DPO 对数的最小分数差异
dpo_ds = dataset.to_dpo_dataset(minimum_gap=0.1)
```

## 数据验证

### 检查转换结果

```python
import json

# 加载转换后的数据
with open("data/collabllm_data.json") as f:
    data = json.load(f)

# 检查结构
first_item = data[0]
print(f"conv_id: {first_item['conv_id']}")
print(f"turns: {len(first_item['turns'])}")
print(f"responses: {len(first_item['turns'][0]['responses'])}")

# 检查数据完整性
for item in data:
    assert "conv_id" in item
    assert "single_turn_prompt" in item
    assert "turns" in item
    assert len(item["turns"]) > 0
    for turn in item["turns"]:
        assert "prompt" in turn
        assert "responses" in turn
        assert len(turn["responses"]) > 0
```

## 常见问题

### Q1: 为什么有些对话没有被转换？

**A:** 可能的原因：
- 对话轮数不足（少于 1 个完整的用户-助手交互）
- 转换过程中出现错误，被日志记录为 "failed"
- 数据格式不符合预期

检查日志输出，查看 "Warning" 或 "Error" 信息。

### Q2: quality_score 应该如何计算？

**A:** 通常由 `DataCleaner` 计算，基于论文中的加权公式：

```
TotalScore = 0.15*SD + 0.10*SV + 0.15*IKT + 0.15*BP + 0.15*SC + 0.10*L3GR + 0.20*3C
```

参见 `collabllm/datasets/cleaner.py` 中的 `calculate_score()` 方法。

### Q3: 如何处理多个不同的教师回复（DPO 训练准备）？

**A:** 使用聚合模式：

```python
# 创建多个相同问题、不同回复的 session
aggregated = convert_sessions_to_nested_with_aggregation(sessions)

# 然后用 MultiturnDataset 加载
dataset = MultiturnDataset(aggregated)

# 转换为 DPO 格式
dpo_ds = dataset.to_dpo_dataset(minimum_gap=0.0)
```

### Q4: 如何自定义元数据？

**A:** 在转换时传递额外的元数据字典：

```python
extra_meta = {
    "model_name": "GPT-4",
    "evaluated_by": "expert_panel",
}

nested = convert_session_to_nested(
    session,
    use_quality_score=True,
    extra_metadata=extra_meta
)
```

## 参考资源

- [MultiturDataset 文档](../collabllm/datasets/multiturn.py)
- [TeachingSession 类定义](../collabllm/datasets/types.py)
- [Converter 模块](../collabllm/datasets/converter.py)
- [使用示例](../examples/convert_examples.py)
- [数据清洗工具](../collabllm/datasets/cleaner.py)
