"""
collabllm.utils.template
~~~~~~~~~~~~~~~~~~~~~~~~
消息模板与格式化辅助函数。

用于将 messages 结构转换为可读文本，并处理系统提示词的裁剪。
"""

def parse_messages(messages, strip_sys_prompt=True):
    """
    Args:
        messages: List[dict]
            包含 role 和 content 的消息列表。
            例如：messages = [{'role': 'user', 'content': 'Hello!'}, ...]
    """
    if messages is None:
        return ""

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(
        f"**{m['role'].capitalize()}**: {m['content']}" for m in messages
    )

    return chat

def strip_system_prompt(messages):
    """
    Args:
        messages: List[dict]
            包含 role 和 content 的消息列表。
            例如：messages = [{'role': 'user', 'content': 'Hello!'}, ...]
    """
    return [msg for msg in messages if msg["role"] != "system"]
