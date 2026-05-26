"""
collabllm.utils.format
~~~~~~~~~~~~~~~~~~~~~~
数据格式判断工具。

该模块提供若干轻量辅助函数，用于判断样本是否属于对话格式等。
"""

from typing import Any


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    判断示例是否为对话（conversational）格式。

    Args:
        example (`dict[str, Any]`):
            数据集中的单条条目。不同数据集可能使用不同的键名。

    Returns:
        `bool`:
            如果数据为对话格式则返回 `True`，否则返回 `False`。

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # 必须包含支持的键之一
    if example_keys:
        key = example_keys.pop()  # 取出一个支持的键
        maybe_messages = example[key]
        # 它应当是一个消息列表（list）
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # 每条消息应为包含 "role" 和 "content" 字段的字典
            if (
                isinstance(maybe_message, dict)
                and "role" in maybe_message
                and "content" in maybe_message
            ):
                return True

    return False
