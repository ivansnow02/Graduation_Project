"""
collabllm.datasets.single_turn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
单轮对话数据的统一加载器与包装器。

支持输入为单轮对话字典列表，并统一转换为 HuggingFace DatasetDict：

1. 输入数据必须包含 `prompt` 和 `completion` 字段。
2. 可选支持 `split` 字段，用于直接指定 train/eval 划分。
3. 若没有 `split` 字段，则按 `eval_ratio` 随机拆分训练集和验证集。

输出字段统一为：
`single_turn_prompt`、`single_turn_completion`、`single_turn_metadata`
"""

from datasets import Dataset, DatasetDict
from typing import List, Dict, Any
import random


class SingleTurnDataset:
    """用于单轮对话数据的 Dataset 封装，兼容 HuggingFace DatasetDict。"""

    def __init__(self, data: List[Dict[str, Any]], eval_ratio: float = 0.1, seed: int = 42):
        """
        初始化 SingleTurnDataset。

        Args:
            data: 单轮对话数据列表。每条数据必须包含 'prompt' 和 'completion'，
                  也可以包含其他元数据字段。
            eval_ratio: 当数据中没有 'split' 字段时，用于划分验证集的比例。
            seed: 随机划分训练/验证集时使用的随机种子。

        Raises:
            ValueError: 当数据为空或缺少必需字段时抛出。
        """
        if not data:
            raise ValueError("Data cannot be empty")

        # 校验必需字段
        required_fields = {"prompt", "completion"}
        self.fields = set(data[0].keys())

        if not required_fields.issubset(self.fields):
            missing = required_fields - self.fields
            raise ValueError(f"Missing required fields: {missing}")

        # 校验所有条目的字段是否一致
        for i, entry in enumerate(data):
            if set(entry.keys()) != self.fields:
                raise ValueError(
                    f"Entry {i} has inconsistent keys. "
                    f"Expected: {self.fields}, Got: {set(entry.keys())}"
                )

        self.data = data
        self.eval_ratio = eval_ratio
        self.seed = seed

    def to_hf_dataset(self) -> DatasetDict:
        """
        将数据转换为 HuggingFace 的 DatasetDict 格式。

        如果数据中存在 'split' 字段，则直接按该字段划分；
        否则按 eval_ratio 随机划分 train/eval。

        Returns:
            包含 train/eval 切分的 DatasetDict，每条样本包含：
            - single_turn_prompt: 输入提示
            - single_turn_completion: 目标回复
            - single_turn_metadata: 其他元数据
        """
        # 如果已有 split 字段，直接使用
        if "split" in self.fields:
            splits = [entry["split"] for entry in self.data]
            unique_splits = list(set(splits))
            split_indices = {
                split: [i for i, x in enumerate(splits) if x == split]
                for split in unique_splits
            }
        else:
            # 否则随机生成 train/eval 划分
            random.seed(self.seed)
            eval_size = int(len(self.data) * self.eval_ratio)
            eval_indices = random.sample(range(len(self.data)), k=min(eval_size, len(self.data)))
            train_indices = list(set(range(len(self.data))) - set(eval_indices))

            split_indices = {
                "train": train_indices,
                "eval": eval_indices,
            }

        # 构建元数据字段（排除 prompt、completion 和 split）
        metadata_fields = self.fields - {"prompt", "completion", "split"}

        dataset_dict = {}
        for split, indices in split_indices.items():
            if not indices:  # 跳过空切分
                continue

            dataset_dict[split] = Dataset.from_dict({
                "single_turn_prompt": [self.data[i]["prompt"] for i in indices],
                "single_turn_completion": [self.data[i]["completion"] for i in indices],
                "single_turn_metadata": [
                    {field: self.data[i][field] for field in metadata_fields}
                    for i in indices
                ]
            })

        return DatasetDict(dataset_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        按索引获取一条数据。

        Args:
            idx: 需要获取的数据索引。

        Returns:
            指定索引的数据项。

        Raises:
            IndexError: 当索引越界时抛出。
        """
        return self.data[idx]

    def __len__(self) -> int:
        """
        Returns:
            数据集样本数量。
        """
        return len(self.data)

    def get_splits_info(self) -> Dict[str, int]:
        """
        返回数据切分信息。

        Returns:
            各切分名称及其样本数。
        """
        if "split" in self.fields:
            splits = [entry["split"] for entry in self.data]
            split_counts = {}
            for split in set(splits):
                split_counts[split] = splits.count(split)
            return split_counts
        else:
            eval_size = int(len(self.data) * self.eval_ratio)
            return {
                "train": len(self.data) - eval_size,
                "eval": eval_size,
            }

