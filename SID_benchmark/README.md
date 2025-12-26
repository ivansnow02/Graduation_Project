
# SID_Benchmark-3CBA

本仓库包含一套用于“跨学科苏格拉底式教学对话”的**数据生成 → 结构化标注 → 客观指标评测 → 主观（裁判模型）评测**的脚本链路。

你当前看到的核心脚本只有 4 个（均位于仓库根目录）：

- `multi_dialogue.py`：生成跨学科多轮师生对话（OpenAI 兼容接口）。
- `annotation.py`：对对话逐轮抽取 9 项教学/认知字段（vLLM + 本地模型）。
- `objective_eval.py`：基于标注结果计算一组“客观”指标并汇总。
- `subjective_eval.py`：使用“裁判模型”对对话做 5 维主观评分，并输出单文件报告与总体报告。

> 说明：仓库中 `datasets/` 下已经包含部分示例数据与已标注结果（`datasets/annotated/` 等）。脚本默认配置通过环境变量（`.env`）提供。

---

## 依赖与环境

项目依赖在 `pyproject.toml` 中声明（Python 版本要求：`>= 3.13`）。推荐使用 `uv` 管理环境。

关键三方依赖：

- `vllm`：用于 `annotation.py` 的本地推理（需要 GPU 环境更合适）。
- `transformers`：加载 tokenizer。
- `openai` + `requests`：用于调用 OpenAI 兼容的 HTTP 接口（`multi_dialogue.py` / `subjective_eval.py`）。
- `tqdm`：主观评测并发时的进度条。

### `.env` 配置（必看）

仓库根目录已提供一个 `.env` 模板文件（不含真实密钥），你需要根据实际服务填写。

常用变量（摘录）：

- 对话生成（`multi_dialogue.py`）
	- `OPENAI_API_KEY`
	- `OPENAI_BASE_URL`（可选；若你的服务不是标准路径，直接用下一项）
	- `OPENAI_CHAT_COMPLETIONS_URL`（推荐：直接填完整 `/chat/completions` 地址）
	- `OUTPUT_DIR`（输出目录）
	- `MULTI_DIALOGUE_INPUT_FILE`（topics 文件路径）

- 标注（`annotation.py`）
	- `ANNOTATION_TOKENIZER`（HuggingFace tokenizer：本地路径或模型名）
	- `ANNOTATION_MODEL`（vLLM 模型：本地路径或模型名）
	- `ANNOTATION_TP_SIZE`（多卡张量并行数，默认 8）
	- `ANNOTATION_INPUT_DIR` / `ANNOTATION_OUTPUT_DIR`

- 主观评测（`subjective_eval.py`）
	- `JUDGE_API_URL`
	- `JUDGE_MODEL`（默认 `deepseek-v3-0324`）
	- `BASE_PATH`（根目录，用于推导默认数据/报告路径）
	- `SUBJECTIVE_DATASET_DIR` / `SUBJECTIVE_FILE_PATTERN` / `SUBJECTIVE_REPORT_DIR`

---

## 数据格式与目录约定

### 1) Topic 文件：跨学科课程内容

典型位置：`datasets/interdisciplinary_topic.json`

格式：一个 JSON 数组，每个元素至少包含：

- `id`：主题 id（建议为字符串或整数）
- `topic`：主题正文（跨学科课程内容/材料）

示例：

```json
[
	{"id": 1, "topic": "……课程材料……"},
	{"id": 2, "topic": "……课程材料……"}
]
```

### 2) 对话数据：`multi_dialogue.py` 输出

输出文件命名（默认）：`multi_dialogue_topic_<topic_id>.json`

文件内容：一个 JSON 数组，每个元素是一段完整对话（同一 topic 下会生成多段对话）。字段：

- `topic_id`：主题 id（来自 topic 文件）
- `topic_text`：主题正文
- `student_id`：例如 `Student_1`
- `student_type`：随机抽样的学生画像标签（如“全优型/知识掌握不足/学习渴望低”）
- `scenario`：随机抽样的情境标签（如“学生不理解问题含义”等）
- `repeat_id`：重复采样编号（例如 `R1` / `R2`）
- `dialogue`：对话轮次数组；每个 turn 结构为：
	- `role`：`"学生"` 或 `"教师"`
	- `content`：该轮文本

示例（截断）：

```json
{
	"topic_id": "topic_1",
	"student_id": "Student_1",
	"repeat_id": "R1",
	"dialogue": [
		{"role": "学生", "content": "……？"},
		{"role": "教师", "content": "……？"}
	]
}
```

### 3) 标注数据：`annotation.py` 输出（JSONL）

`annotation.py` 会读取某个 `*.json`（对话数组），然后输出为 `*.jsonl`：

- 每行一个 conversation（一个 JSON 对象）
- 在原始 entry 上新增字段 `annotations`

`annotations` 是一个列表，列表元素为逐轮标注结果（每个 turn 一条）。每条包含 9 个字段（见下文）。

---

## 脚本逐个详解（源码级）

### `multi_dialogue.py`：生成多轮苏格拉底式对话

#### 目标

给定一个跨学科主题材料（topic_text），模拟：

1) 学生提出一个“真正困惑/好奇”的初始问题。
2) 教师**只用苏格拉底式追问**逐轮引导（每轮只能一个问题，不直接给答案）。
3) 学生基于不同画像与情境作答。
4) 当达到一定轮数或教师判断理解完成后，教师生成总结并用 `[结束]` 标记。

#### 关键函数

- `call_gpt_segmented(...)`
	- 用 `requests.post(...)` 调用 OpenAI 兼容接口。
	- “分段续写”机制：如果输出可能很长，会用“（请继续上一轮内容…）”提示模型继续，最多 `max_segments` 段。
	- 截断策略：当 `len(content) >= max_tokens` 时，尝试在最近的 `。/？` 处截断。
	- 终止条件：出现 `[结束]` 或输出明显变短（`len(content) < max_tokens*0.8`）

- `generate_initial_student_question(topic_text)`
	- 让模型扮演中学生，对课程材料提出**仅一个问题**。
	- 末尾保证带 `？`。

- `generate_teacher_response(history_text)`
	- 教师角色：强制苏格拉底式提问法。
	- 约束：每轮只能一个简洁问题；鼓励跨学科；当完成理解则输出总结并标注 `[结束]`。

- `generate_student_response(history, ...)`
	- 随机抽样学生画像（全优/基础薄弱/兴趣不高）+ 随机场景。
	- 要求回答简洁、直接、避免过度推理（用于模拟不同质量学生反应）。

- `generate_full_dialogue(topic_text, student_id, min_turns=3)`
	- 循环生成教师/学生轮次。
	- 当 `turns > 5` 时强制生成一次总结并结束（即：最长大约 6 轮教师引导 + 总结）。

- `generate_single_topic_dialogues((topic_entry, topic_idx, base_output_dir))`
	- 每个 topic：生成 20 个学生 × 2 次 repeat = 40 段对话。
	- 输出为 `multi_dialogue_topic_<topic_id>.json`（数组）。

#### 输入 / 输出

- 输入：`MULTI_DIALOGUE_INPUT_FILE` 指向的 topics JSON。
- 输出：`OUTPUT_DIR/multi_dialogue_topic_<topic_id>.json`

#### 运行要点

1) `.env` 中必须配置：`OPENAI_API_KEY` +（`OPENAI_CHAT_COMPLETIONS_URL` 或 `OPENAI_BASE_URL`）。
2) `__main__` 中通过环境变量读取：
	 - `MULTI_DIALOGUE_INPUT_FILE`
	 - `OUTPUT_DIR`
3) 并行：默认 `multiprocessing.Pool`，进程数 `min(10, cpu_count())`。

---

### `annotation.py`：逐轮抽取 9 项结构化标注（vLLM）

#### 目标

将师生对话转为结构化教学标注，字段固定为 9 项，并以 **JSON 列表**输出（每轮一条）。

#### 标注字段（prompt 中定义）

每个 turn 输出一个对象：

1. `speaker`：`"教师"` / `"学生"`
2. `utterance`：原始发言
3. `teacher_intent`：教师意图（仅教师轮）
	 - 取值之一：`引出概念 / 检测理解 / 引导推理 / 引发迁移 / 总结提升`
4. `teaching_strategy`：教师策略（仅教师轮）
	 - 例如：追问、提示、类比、情境设问、拆解问题、鼓励回应、正误反馈…
5. `discipline`：学科（允许多学科，用逗号分隔）
6. `discipline_transfer`：是否相对上轮出现新的学科并引导联系（`是/否`）
7. `student_cognition_state`：学生认知状态（仅学生轮）
	 - 取值之一：`清晰理解 / 模糊理解 / 表达困难 / 答非所问 / 错误回答 / 高阶思考`
8. `teacher_guidance_level`：教师引导层级（仅教师轮）
	 - L1：封闭性问题；L2：解释/理解型；L3：迁移/推理/综合
9. `cognitive_level`：Bloom 层级
	 - `记忆 / 理解 / 应用 / 分析 / 评价 / 创造`

#### 关键函数

- `build_prompt(dialogue_pair)`
	- 将“两轮对话（通常教师+学生或学生+教师）”拼成一个标注任务。
	- 强约束输出为 JSON 数组。

- `extract_first_json_array(s)`
	- 从模型输出中提取第一段 `[...]`（通过括号深度计数保证闭合）。
	- 目的：尽量从带有杂质文本的输出中“捞出” JSON。

- `annotate_pair(dialogue_pair)`
	- 将 prompt 喂给 vLLM，解析输出的 JSON 数组。

- `annotate_dialogue_file(input_path, output_path)`
	- 读入一个 `*.json`（数组），输出一个 `*.jsonl`。
	- **断点续跑**：`load_existing_results` 会统计已有输出行数，并从下一条继续。
	- 对每段对话：按 `turns[0:2]`, `turns[2:4]`… 两两成对进行标注，累积到 `annotations`。

- `batch_process(input_dir, output_dir, input_pattern='*.json')`
	- 批量处理目录下所有匹配文件；输出同名 `.jsonl`。

#### 输入 / 输出

- 输入：目录下 `*.json`（每个文件为对话数组；每个元素含 `dialogue` 字段）。
- 输出：目录下 `*.jsonl`（每行一段对话，含新增 `annotations` 字段）。

#### 运行要点（配置）

`annotation.py` 通过环境变量读取：

- `ANNOTATION_TOKENIZER`
- `ANNOTATION_MODEL`
- `ANNOTATION_TP_SIZE`（默认 8）
- `ANNOTATION_INPUT_DIR` / `ANNOTATION_OUTPUT_DIR`

---

### `objective_eval.py`：客观指标计算与汇总

#### 目标

读取带 `annotations` 的 JSONL，对每段对话计算一组指标，并输出：

1) 每个输入文件一个 `*_metrics.json`
2) 一个总体汇总 `_overall_summary_metrics.json`（可通过参数指定）

#### 指标定义（代码严格实现）

记号：

- $T$：教师轮数（`speaker == '教师'` 的标注条数）
- $S$：学生轮数

1) **StrategyDensity（策略密度）**

教师轮中 `teaching_strategy` 非空的占比：
$$\text{StrategyDensity} = \frac{\#\{t\in T: strategy(t)\neq \emptyset\}}{|T|}$$

2) **StrategyVariety（策略多样性）**

教师使用到的“不同策略种类数 / 8”。其中 8 来自常量 `TOTAL_POSSIBLE_STRATEGIES = 8`：
$$\text{StrategyVariety} = \frac{|\{strategy(t)\}|}{8}$$

3) **IKT（跨学科迁移率）**

代码中统计 `discipline_transfer == '是'` 的标注条数，然后除以教师轮数：
$$\text{IKT} = \frac{\#\{a: a.discipline\_transfer='是'\}}{|T|}$$

> 注意：分子是对全体 `annotations` 统计（包含学生轮），分母是教师轮数。若你希望“仅在教师轮统计迁移”，需要自行调整源码。

4) **BP（Bloom Progression，认知进阶）**

对学生轮的 Bloom 层级映射为 1~6（`BLOOM_MAP`），取最大值减最小值得到 raw progression：

$$\text{BP}_{raw} = \max(level_S) - \min(level_S)$$

并归一化到 0~1（最大可能跳跃为 $6-1=5$）：
$$\text{BP} = \frac{\text{BP}_{raw}}{5}$$

5) **StructureCompleteness（结构完整性）**

教师意图覆盖率。要求集合为：

`REQUIRED_INTENTS = {引出概念, 引导推理, 引发迁移, 总结提升}`（4 类，不含“检测理解”）。

$$\text{StructureCompleteness} = \frac{|Intents\_{teacher} \cap Required|}{4}$$

6) **L3GuidanceRate（L3 引导率）**

教师轮中 `teacher_guidance_level == 'L3'` 的占比。

7) **CognitiveCorrectionRate（认知纠错率 / 3C）**

统计学生轮的 `student_cognition_state == '错误回答'` 次数；如果后继一轮学生状态变为 `高阶思考` 或 `清晰理解`，记为一次成功纠错。

- 若错误数为 0，则该指标直接记为 1.0。
- 否则：成功纠错 / 错误数。

8) **TotalScore（总分）**

加权线性组合（源码权重）：

$$
\begin{aligned}
	ext{TotalScore}=&\ 0.15\cdot StrategyDensity + 0.10\cdot StrategyVariety + 0.15\cdot IKT\\
&+ 0.15\cdot BP + 0.15\cdot StructureCompleteness + 0.10\cdot L3GuidanceRate\\
&+ 0.20\cdot CognitiveCorrectionRate
\end{aligned}
$$

#### 命令行用法

`objective_eval.py` 使用 `argparse`，支持传入多个文件（含通配符，由 shell 展开）：

```bash
python objective_eval.py datasets/annotated/EduChat-R1/*.jsonl \
	--summary-file outputs/_overall_summary_metrics.json
```

输出：

- 每个输入文件产生：`<base_name>_metrics.json`（写在 `--summary-file` 所在目录）
- 最终汇总：`--summary-file` 指定路径

---

### `subjective_eval.py`：裁判模型主观评分

#### 目标

让裁判模型对每段对话按 5 个指标打 1~5 分，并给出不超过 50 字理由；最终输出：

- 单文件评测报告：包含平均分、分数分布、逐对话明细
- 总体报告：跨文件汇总（按对话数加权）

#### 五个主观指标（prompt 中定义）

1. `X-SRG`：跨学科脚手架引导评分（越“循序渐进的追问引导”越高）
2. `M-RCC`：多学科推理链条完整性（A→B→C 链条清晰）
3. `X-MSR`：跨学科错误迁移识别与修复
4. `CTRA`：跨学科推理连接（迁移是否自然）
5. `TCF`：学科过渡流畅度

#### 关键函数

- `build_evaluation_prompts(dataset)`
	- 将每段对话拼成评测 prompt。
	- 强制裁判输出为**纯 JSON**，禁止代码块。

- `generate_with_api(prompt, max_retries=3)`
	- 调用 `JUDGE_API_URL`（OpenAI 兼容）请求评测。
	- `stop` 中设置了多种截断标记，尽量避免输出溢出。

- `parse_model_response(response)`
	- 解析裁判输出：
		- 先尝试直接 `json.loads`
		- 再尝试去掉代码块围栏
		- 再用正则提取 `{...}` 片段
	- 若缺失字段则补默认：`{"score": null, "reason": "解析失败"}`

- `evaluate_dialogues(dataset, file_num, max_workers=4)`
	- 使用 `ThreadPoolExecutor` 并发评测每条对话。

- `generate_single_report(results, file_num)`
	- 统计平均分与分布，落盘 JSON。

- `generate_overall_report(all_reports)`
	- 对全文件做按对话数加权汇总。

#### 输入 / 输出

- 输入：一组 `*.json` 对话文件（即 `multi_dialogue.py` 输出的数组文件）。
- 文件匹配：默认 `SUBJECTIVE_FILE_PATTERN = <dataset_dir>/**/*_topic_*.json`。
- 输出：默认写入
	- `reports/subjective/<judge_model_prefix>/eval_topic<file_num>_<judge_model_prefix>.json`
	- `reports/subjective/<judge_model_prefix>/overall_eval_<judge_model_prefix>.json`

其中 `<judge_model_prefix>` 是 `JUDGE_MODEL.split('-')[0]`（例如 `deepseek`）。

---

## 推荐的端到端流程（从生成到评测）

1) 准备 topics：例如 `datasets/interdisciplinary_topic.json`
2) 生成对话：运行 `multi_dialogue.py`（输出多个 `multi_dialogue_topic_*.json`）
3) 批量标注：运行 `annotation.py`（将对话 JSON 转成带 `annotations` 的 JSONL）
4) 客观指标：运行 `objective_eval.py` 对 JSONL 计算指标并汇总
5) 主观指标：运行 `subjective_eval.py` 让裁判模型评分并输出报告

---

## 常见问题（排障提示）

### 1) 标注阶段 JSON 解析失败

`annotation.py` 依赖模型严格输出 JSON 数组。若失败：

- 优先检查模型是否会输出多余解释（可调低温度或强化 prompt 约束）。
- 观察日志：`Failed to extract JSON array...` / `JSON Failed...`。
- 适当降低 `max_tokens` 或在模型端加 JSON 模式/工具调用（如你使用的模型支持）。

### 2) 主观评测解析失败

`subjective_eval.py` 已做了三层解析兜底，但裁判模型仍可能输出非 JSON。

- 确认你的裁判服务确实返回 `choices[0].message.content`。
- 可调 `temperature` 更低（当前 0.1）。
- 若模型经常输出多余文本，可增加 `stop` 列表或在 prompt 中更强约束。

### 3) 为什么 IKT 的分子统计包含学生轮？

这是当前实现细节：

- 分子：对所有 `annotations` 统计 `discipline_transfer == '是'`
- 分母：教师轮数

若你希望严格“教师引导带来的迁移”，可在 `objective_eval.py` 中将分子限定为教师轮。

---

## 文件一览（源码定位）

- `multi_dialogue.py`：对话生成（OpenAI 兼容 HTTP）
- `annotation.py`：vLLM 标注（9 字段）
- `objective_eval.py`：客观指标（含总分加权公式）
- `subjective_eval.py`：主观评分（5 维，带并发与报告）
- `.env`：运行配置模板（请填写真实值）
