"""从输入字符串中提取并解析第一个 JSON 对象或数组。

函数会在包含额外文本的响应中定位第一个左大括号或左中括号，裁剪并以一个
轻量级解析器解析为 Python 对象（支持数字自动转换、三引号字符串等）。
"""


def extract_json(s):
    idx_brace = s.find("{")
    idx_bracket = s.find("[")

    if idx_brace == -1 and idx_bracket == -1:
        raise ValueError("No JSON object or array found")

    if idx_brace != -1 and (idx_bracket == -1 or idx_brace < idx_bracket):
        # 检测到 JSON 对象（以 '{' 开始）
        json_start = idx_brace
        json_end = s.rfind("}")
    else:
        # 检测到 JSON 数组（以 '[' 开始）
        json_start = idx_bracket
        json_end = s.rfind("]")

    if json_end == -1:
        raise ValueError("No closing brace/bracket found")

    s = s[json_start : json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f"Unexpected content at position {pos}")
    return result


def parse_value(s, pos):
    pos = skip_whitespace(s, pos)
    if pos >= len(s):
        raise ValueError("Unexpected end of input")
    if s[pos] == "{":
        return parse_object(s, pos)
    elif s[pos] == "[":
        return parse_array(s, pos)
    elif s[pos : pos + 3] in ("'''", '"""'):
        return parse_triple_quoted_string(s, pos)
    elif s[pos] in ('"', "'"):
        return parse_string(s, pos)
    elif s[pos : pos + 4].lower() == "true":
        return True, pos + 4
    elif s[pos : pos + 5].lower() == "false":
        return False, pos + 5
    elif s[pos : pos + 4].lower() == "null":
        return None, pos + 4
    elif s[pos] in "-+0123456789.":
        return parse_number(s, pos)
    else:
        raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")


def parse_object(s, pos):
    obj = {}
    assert s[pos] == "{"
    pos += 1
    pos = skip_whitespace(s, pos)
    while pos < len(s) and s[pos] != "}":
        pos = skip_whitespace(s, pos)
        key, pos = parse_key(s, pos)
        pos = skip_whitespace(s, pos)
        if pos >= len(s) or s[pos] != ":":
            raise ValueError(f'Expected ":" at position {pos}')
        pos += 1
        pos = skip_whitespace(s, pos)
        value, pos = parse_value(s, pos)
        obj[key] = value
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ",":
            pos += 1
            pos = skip_whitespace(s, pos)
        elif pos < len(s) and s[pos] == "}":
            break
        elif pos < len(s) and s[pos] != "}":
            raise ValueError(f'Expected "," or "}}" at position {pos}')
    if pos >= len(s) or s[pos] != "}":
        raise ValueError(f'Expected "}}" at position {pos}')
    pos += 1
    return obj, pos


def parse_array(s, pos):
    lst = []
    assert s[pos] == "["
    pos += 1
    pos = skip_whitespace(s, pos)
    while pos < len(s) and s[pos] != "]":
        value, pos = parse_value(s, pos)
        lst.append(value)
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ",":
            pos += 1
            pos = skip_whitespace(s, pos)
        elif pos < len(s) and s[pos] == "]":
            break
        elif pos < len(s) and s[pos] != "]":
            raise ValueError(f'Expected "," or "]" at position {pos}')
    if pos >= len(s) or s[pos] != "]":
        raise ValueError(f'Expected "]" at position {pos}')
    pos += 1
    return lst, pos


def parse_string(s, pos):
    quote_char = s[pos]
    assert quote_char in ('"', "'")
    pos += 1
    result = ""
    while pos < len(s):
        c = s[pos]
        if c == "\\":
            pos += 1
            if pos >= len(s):
                raise ValueError("Invalid escape sequence")
            c = s[pos]
            escape_sequences = {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "\\": "\\",
                quote_char: quote_char,
            }
            result += escape_sequences.get(c, c)
        elif c == quote_char:
            pos += 1
            # 尝试将字符串内容转换为数字（如果可能的话）
            converted_value = convert_value(result)
            return converted_value, pos
        else:
            result += c
        pos += 1
    raise ValueError("Unterminated string")


def parse_triple_quoted_string(s, pos):
    if s[pos : pos + 3] == "'''":
        quote_str = "'''"
    elif s[pos : pos + 3] == '"""':
        quote_str = '"""'
    else:
        raise ValueError(f"Expected triple quotes at position {pos}")
    pos += 3
    result = ""
    while pos < len(s):
        if s[pos : pos + 3] == quote_str:
            pos += 3
            # 尝试将三引号字符串内容转换为数字（如可能）
            converted_value = convert_value(result)
            return converted_value, pos
        else:
            result += s[pos]
            pos += 1
    raise ValueError("Unterminated triple-quoted string")


def parse_number(s, pos):
    start = pos
    while pos < len(s) and s[pos] in "-+0123456789.eE":
        pos += 1
    num_str = s[start:pos]
    try:
        if "." in num_str or "e" in num_str.lower():
            return float(num_str), pos
        else:
            return int(num_str), pos
    except ValueError:
        raise ValueError(f"Invalid number at position {start}: {num_str}")


def parse_key(s, pos):
    pos = skip_whitespace(s, pos)
    if s[pos] in ('"', "'"):
        key, pos = parse_string(s, pos)
        return key, pos
    else:
        raise ValueError(f"Expected string for key at position {pos}")


def skip_whitespace(s, pos):
    while pos < len(s) and s[pos] in " \t\n\r":
        pos += 1
    return pos


def convert_value(value):
    true_values = {"true": True, "false": False, "null": None}
    value_lower = value.lower()
    if value_lower in true_values:
        return true_values[value_lower]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value  # 如果不能转换为数字，则作为字符串返回
