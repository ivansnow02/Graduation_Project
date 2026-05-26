"""
collabllm.datasets.multiturn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
多轮对话数据的统一加载器与包装器。

初始化支持三种输入形式：

- 扁平列表 (List[dict])，每行包含必需字段：
     {'prompt', 'completion', 'conv_id', 'score',
        'single_turn_prompt', 'single_turn_completion', 'single_turn_metadata'}

- 嵌套结构（每个对话为一个 dict）：
     参考文件中常见的嵌套格式，包含 conv_id、turns 等字段。

- 本地 HF 数据集目录（Dataset.save_to_disk 保存的目录）或 HF Hub 仓库 ID（字符串）。

内部统一将数据转换为扁平的 self.data（List[dict]），包含字段：
{prompt, completion, conv_id, score, single_turn_prompt, single_turn_completion,
 single_turn_metadata, turn_id}

派生字段说明
-------------
- 若未显式给出，`turn_id` 将被设置为 `len(prompt)`。

转换器（均使用均匀随机拆分）
--------------------------------
- `to_sft_dataset()` -> DatasetDict {messages}
- `to_dpo_dataset()` -> DatasetDict {prompt, chosen, rejected, score_*}
- `to_inputs_dataset()` -> DatasetDict {prompt, single_turn_*}
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from collabllm.prompts import SYSTEM_PROMPT
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

_REQUIRED: set[str] = {
    "prompt",
    "completion",
    "conv_id",
    "score",
    "single_turn_prompt",
    "single_turn_completion",
    "single_turn_metadata",
}

import logging

logger = logging.getLogger(__name__)


# 均匀拆分器
def _uniform_split(
    full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict({
        "train": full_ds.select(train_idx),
        "eval": full_ds.select(sorted(eval_idx)),
    })


# 主数据类
class MultiturnDataset:
    def __init__(
        self,
        data_or_local_dir_or_hf_repo_or_nested: Union[List[Dict[str, Any]], str],
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        """
        Args:
            data_or_local_dir_or_hf_repo_or_nested:
                扁平字典列表（旧格式），或嵌套对话列表（新格式），或本地由
                Dataset.save_to_disk 保存的目录路径，或 HF Hub 仓库 ID。
            seed:
                用于均匀拆分的随机种子。
            add_system_prompt:
                是否在每条样本前添加系统消息。
        """
        self.seed = seed
        self.sys_msg = (
            [{"role": "system", "content": SYSTEM_PROMPT}] if add_system_prompt else []
        )

        # 加载原始数据到字典的 raw_list 中
        if isinstance(data_or_local_dir_or_hf_repo_or_nested, list):
            raw_list = data_or_local_dir_or_hf_repo_or_nested
        elif os.path.exists(str(data_or_local_dir_or_hf_repo_or_nested)):
            path = str(data_or_local_dir_or_hf_repo_or_nested)
            if os.path.isdir(path):
                ds_dict = load_from_disk(path)  # type: ignore
                # 如果是 DatasetDict，可能需要特殊处理；原始代码假设其支持 .flatten()
                if hasattr(ds_dict, "flatten"):
                    raw_list = [dict(r) for r in ds_dict.flatten()]  # type: ignore
                else:
                    # 如果是 DatasetDict，合并所有 split
                    raw_list = [dict(r) for split in ds_dict.values() for r in split]  # type: ignore
            else:
                # 处理 JSON / JSONL 文件
                import json

                if path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        raw_list = json.load(f)
                elif path.endswith(".jsonl"):
                    raw_list = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                raw_list.append(json.loads(line))
                else:
                    # 其他文件类型回退使用 load_dataset 加载
                    ds = load_dataset("json", data_files=path, split="train")
                    raw_list = [dict(r) for r in ds]
        else:
            ds_dict = load_dataset(
                str(data_or_local_dir_or_hf_repo_or_nested), trust_remote_code=True
            )  # type: ignore
            raw_list = [dict(r) for _, split in ds_dict.items() for r in split]

        # 过滤掉可能由转换失败产生的 None 条目
        raw_list = [row for row in raw_list if row is not None]

        if not raw_list:
            raise ValueError(
                "Loaded dataset is empty (or contains only empty entries)."
            )

        # 检测是否为嵌套结构：检查第一个元素是否包含 "turns" 键
        if isinstance(raw_list[0], dict) and "turns" in raw_list[0]:
            self.data = self._flatten_nested(raw_list)
        elif isinstance(raw_list[0], dict):
            # 视为扁平结构；验证必需字段
            if not _REQUIRED.issubset(raw_list[0]):
                missing = _REQUIRED - set(raw_list[0])
                raise ValueError(f"Missing required keys in flat data: {missing}")

            # 如果缺失 turn_id 则自动填充
            for row in raw_list:
                if not isinstance(row.get("prompt"), Sequence):
                    raise TypeError(
                        f"Row {row.get('conv_id')} 的 `prompt` 必须为消息列表。当前类型: {type(row.get('prompt'))}"
                    )
                row.setdefault("turn_id", len(row["prompt"]))

            self.data = raw_list  # type: ignore
        else:
            raise TypeError(
                f"Unknown data type in raw_list: {type(raw_list[0])}. Expected dict."
            )

        if not self.data:
            raise ValueError("No valid rows after processing input.")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: Optional[str] = None,
        split: Optional[str] = None,
    ) -> DatasetDict:
        """
        Push the dataset to the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository ID on the Hugging Face Hub.
        private : bool
            Whether to create a private repository.
        token : Optional[str]
            Optional authentication token for the Hub.
        split : Optional[str]
            If provided, will save only this split (e.g., "train", "eval").

        Returns:
        -------
        DatasetDict
            The pushed dataset.
        """
        ds = Dataset.from_dict({k: [row[k] for row in self.data] for k in self.data[0]})
        return ds.push_to_hub(repo_id, private=private, token=token, split=split)

    def _flatten_nested(self, nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert nested conversation format to flat list of rows.

        Nested format per conversation:
        {
          "conv_id": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turns": [
             {
               "prompt": [...],
               "responses": [
                  {"completion": ..., "score": ..., **kwargs}, ...
               ]
             },
             ...
          ]
        }

        Output per row:
        {
          "prompt": [...],
          "completion": ...,
          "conv_id": ...,
          "score": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turn_id": len(prompt)
        }
        """
        flat = []
        for base_conv_id, convo in enumerate(nested):
            # 验证必需的会话级别键是否存在
            for key in {
                "single_turn_prompt",
                "single_turn_completion",
                "single_turn_metadata",
                "turns",
            }:
                if key not in convo:
                    raise ValueError(f"Missing key '{key}' in nested conversation.")
            st_prompt = convo["single_turn_prompt"]
            st_completion = convo["single_turn_completion"]
            st_metadata = convo["single_turn_metadata"]

            for turn in convo["turns"]:
                if "prompt" not in turn or "responses" not in turn:
                    raise ValueError("Each turn must have 'prompt' and 'responses'.")
                prompt_msgs = turn["prompt"]
                if not isinstance(prompt_msgs, Sequence):
                    raise TypeError("`turn['prompt']` must be a list of messages.")
                turn_id = len(prompt_msgs)
                for resp in turn["responses"]:
                    if "completion" not in resp or "score" not in resp:
                        raise ValueError(
                            "Each response must have 'completion' and 'score'."
                        )
                    flat.append({
                        "prompt": prompt_msgs,
                        "completion": resp["completion"],
                        "conv_id": base_conv_id,
                        "score": resp["score"],
                        "single_turn_prompt": st_prompt,
                        "single_turn_completion": st_completion,
                        "single_turn_metadata": st_metadata,
                        "turn_id": turn_id,
                        **{
                            k: resp.get(k)
                            for k in resp
                            if k not in {"completion", "score"}
                        },
                    })
        return flat

    # SFT（监督微调）
    def to_sft_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
        lower_bound_metric: Optional[str] = None,
        lower_bound: Optional[float] = 0.0,
    ) -> DatasetDict:
        # 为每个对话 ID 选择最佳样本：优先选择最新轮次，其次选择分数更高者
        best_examples = {}
        for row in self.data:
            cid = row["conv_id"]
            prev = best_examples.get(cid)
            if (
                prev is None
                or row["turn_id"] > prev["turn_id"]
                or (row["turn_id"] == prev["turn_id"] and row["score"] > prev["score"])
            ):
                best_examples[cid] = row

        # 构建 SFT 对话，按可选的指标阈值进行过滤
        serialized_dialogues = []
        for row in best_examples.values():
            if lower_bound_metric:
                try:
                    metric = row
                    for key in lower_bound_metric.split("."):
                        metric = metric.get(key, {})
                    value = np.asarray(metric).mean().item()
                except Exception as e:
                    logger.error(
                        f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}"
                    )
                    continue

                if value < lower_bound:
                    logger.warning(
                        f"Filtered out conv_id={row['conv_id']} (turn_id={row['turn_id']}) "
                        f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                    )
                    continue

            if not isinstance(row["prompt"], list):
                raise TypeError("Expected `prompt` to be a list of messages.")

            messages = (
                self.sys_msg
                + row["prompt"]
                + [{"role": "assistant", "content": row["completion"]}]
            )
            serialized_dialogues.append(messages)

        logger.info(
            f"Converted {len(serialized_dialogues)} dialogues "
            f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
            f"retention ratio: {len(serialized_dialogues) / len(best_examples):.2f}"
        )

        full_dataset = Dataset.from_dict({"messages": serialized_dialogues})
        return _uniform_split(
            full_dataset, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed
        )

    # DPO（偏好对比优化）
    def to_dpo_dataset(
        self,
        *,
        minimum_gap: float = 0.0,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
    ) -> DatasetDict:
        # 按 (conv_id, turn_id) 分组
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in self.data:
            grouped.setdefault((r["conv_id"], r["turn_id"]), []).append(r)

        pairs = []
        for items in grouped.values():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda r: r["score"], reverse=True)
            best = items[0]
            for j in range(1, len(items)):
                rejected = items[j]
                margin = best["score"] - rejected["score"]
                if margin < minimum_gap:
                    continue
                pairs.append({
                    "prompt": self.sys_msg + best["prompt"],
                    "chosen": best["completion"],
                    "rejected": rejected["completion"],
                    "score_chosen": best["score"],
                    "score_rejected": rejected["score"],
                    "margin": margin,
                })

        logger.info(
            f"Converted {len(pairs)} pairs (minimum_gap={minimum_gap}, ratio={len(pairs) / len(self.data):.2f})"
        )

        if not pairs:
            return DatasetDict({
                "train": Dataset.from_dict({}),
                "eval": Dataset.from_dict({}),
            })

        full_ds = Dataset.from_dict({k: [p[k] for p in pairs] for k in pairs[0]})
        return _uniform_split(
            full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed
        )

    # Inputs（输入集）
    def to_inputs_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
    ) -> DatasetDict:
        # 每个 (conv_id, turn_id) 只保留一行记录
        unique: Dict[tuple, Dict[str, Any]] = {}
        for r in self.data:
            key = (r["conv_id"], r["turn_id"])
            if key not in unique:
                unique[key] = r
            r["prompt"] = self.sys_msg + r["prompt"]

        keep_keys = [
            "prompt",
            "single_turn_prompt",
            "single_turn_completion",
            "single_turn_metadata",
        ]
        records = [{k: row[k] for k in keep_keys} for row in unique.values()]
        if not records:
            return DatasetDict({
                "train": Dataset.from_dict({}),
                "eval": Dataset.from_dict({}),
            })

        full_ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in keep_keys})
        return _uniform_split(
            full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed
        )

    # 其他工具方法（misc）
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
