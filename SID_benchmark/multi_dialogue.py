import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import requests
from openai import OpenAI
import dotenv

dotenv.load_dotenv()


def _join_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    path = (path or "").lstrip("/")
    if not base:
        return ""
    return f"{base}/{path}"


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "").strip()
OUTPUT_FILE = (
    os.path.join(OUTPUT_DIR, "dialogue.jsonl") if OUTPUT_DIR else "dialogue.jsonl"
)

# ===== 教师配置（LM Studio 本地） =====
TEACHER_BASE_URL = os.getenv("TEACHER_BASE_URL", "").strip()
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY", "").strip()
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gpt-4").strip()
teacher_url = os.getenv("TEACHER_CHAT_COMPLETIONS_URL", "").strip()
if not teacher_url and TEACHER_BASE_URL:
    teacher_url = _join_url(TEACHER_BASE_URL, "v1/chat/completions")

# ===== 学生配置（远程 API） =====
STUDENT_OPENAI_API_KEY = os.getenv("STUDENT_OPENAI_API_KEY", "").strip()
STUDENT_OPENAI_BASE_URL = os.getenv("STUDENT_OPENAI_BASE_URL", "").strip()
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "gpt-4").strip()
student_client = OpenAI(
    base_url=STUDENT_OPENAI_BASE_URL or None, api_key=STUDENT_OPENAI_API_KEY or None
)
student_url = os.getenv("STUDENT_OPENAI_CHAT_COMPLETIONS_URL", "").strip()
if not student_url and STUDENT_OPENAI_BASE_URL:
    student_url = _join_url(STUDENT_OPENAI_BASE_URL, "v1/chat/completions")

# 学生 API 请求头
student_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {student_client.api_key}",
}

# 教师 API 请求头
teacher_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TEACHER_API_KEY}",
}


def call_api_segmented(
    prompt,
    url,
    headers,
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    max_segments=6,
):
    """通用 API 调用函数"""
    replies = []
    continuation_prompt = "（请继续上一轮内容，继续输出，不要重复开头）"
    for i in range(max_segments):
        seg_prompt = prompt if i == 0 else prompt + "\n" + continuation_prompt
        data = {
            "model": model,
            "messages": [{"role": "user", "content": seg_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            if not url:
                raise RuntimeError(
                    "Missing chat completions URL. Please configure .env."
                )
            if not headers.get("Authorization"):
                raise RuntimeError("Missing API key. Please configure .env.")
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    # 检查响应中是否有 choices 字段
                    if "choices" not in response_data:
                        print(
                            f"API response missing 'choices' field. Full response: {response.text}"
                        )
                        print(f"Check model name: {model}")
                        replies.append("[Call Failed]")
                        break

                    content = response_data["choices"][0]["message"]["content"].strip()

                    if len(content) >= max_tokens:
                        split_index = max(content.rfind("。"), content.rfind("？"))
                        if split_index != -1:
                            content = content[: split_index + 1]
                    replies.append(content)
                    if "[结束]" in content or len(content) < int(max_tokens * 0.8):
                        break
                except (KeyError, IndexError, ValueError) as json_err:
                    print(f"Error parsing API response: {json_err}")
                    print(f"Full response: {response.text}")
                    replies.append("[Call Failed]")
                    break
            else:
                print(f"API call failed, status code:{response.status_code}")
                print(f"Response: {response.text}")
                replies.append("[Call Failed]")
                break
        except Exception as e:
            print(f"The call failed with the error message: {e}")
            replies.append("[Call Failed]")
            break
    return "\n".join(replies)


def call_gpt_segmented(
    prompt, model=None, temperature=0.7, max_tokens=1000, max_segments=6
):
    """教师 API 调用"""
    if model is None:
        model = TEACHER_MODEL
    return call_api_segmented(
        prompt,
        teacher_url,
        teacher_headers,
        model,
        temperature,
        max_tokens,
        max_segments,
    )


def generate_initial_student_question(topic_text):
    prompt = f"""
你是一名中学生，正在学习一节跨学科课程。请根据下面的课程内容，提出一个你真正感到困惑或好奇的问题。
你的输出中只能包含一个清晰的问题，不得包含多个问句或并列问题。
课程内容如下：
{topic_text}
"""
    result = call_gpt_segmented(prompt)
    result = result.strip().replace("学生：", "").replace("老师：", "").strip()
    return result.split("？")[0].strip() + "？" if "？" in result else result


def generate_teacher_response(history_text):
    prompt = f"""
你是一位跨学科的教师，始终使用苏格拉底式提问法来引导学生。你的目标不是直接给出答案，而是通过逐轮递进、循序渐进的问题，引导学生独立思考并构建跨学科理解。
你的行为规范如下：
每一轮只能提出一个简洁的问题；必须体现认知推进；不能重复提问；禁止多个并列问题。
你应从学生上一次回答中提炼关键点，沿着一个核心问题主线继续深入。
你应始终鼓励学生跨学科思考（生物、地理、物理、历史等）。
当你判断学生理解已经完成，请生成一句简洁明了的总结，明确指出学生已经完成推理，并标注[结束]。
以下是对话历史：
{history_text}
教师：
"""
    result = call_gpt_segmented(prompt)
    result = result.strip().replace("教师：", "").strip()
    return result.split("？")[0].strip() + "？" if "？" in result else result


def generate_student_response(history, max_retries=3, temperature=0.5):
    student_types = {
        "全优型": "你是一位全优型学生，逻辑清晰，善于跨学科推理。",
        "知识掌握不足": "你是一位基础薄弱的学生，对概念掌握不牢。",
        "学习渴望低": "你是一位兴趣不高的学生，对问题有些迷茫但愿意尝试。请用简短、真实的语言回应老师的问题。",
    }
    scenarios = [
        "（1）学生不理解问题的含义",
        "（2）学生不理解教师讲解的内容",
        "（3）学生计算错误",
        "（4）某学生知识掌握较差",
        "（5）学生求知欲弱",
        "（6）学生各方面能力都很强",
    ]
    identity, instruction = random.choice(list(student_types.items()))
    scenario = random.choice(scenarios)
    history_text = "\n".join([f"{h['role']}：{h['content']}" for h in history])
    prompt = f"""
你是一名中学生，现在将模拟一次跨学科教学对话。请根据教师的问题给出简洁、直接的回答，避免过度推理，不进行深度思考，确保回答适合你的理解水平。
学生身份：{identity}
{instruction}
当前情景：{scenario}
以下是对话历史：
{history_text}
学生：
"""
    result = call_api_segmented(
        prompt,
        student_url,
        student_headers,
        model=STUDENT_MODEL,
        temperature=temperature,
    )
    return result, identity, scenario


def generate_summary_from_history(history):
    history_text = "\n".join([f"{h['role']}：{h['content']}" for h in history])
    prompt = f"""
你是一位跨学科教师，请根据以下对话历史，总结这段师生对话并根回答学生最初提出的问题。请用简洁自然的语言总结，注意不要过长，并以[结束]结尾。请确保总结是以教师的口吻对学生进行反馈，而非单纯的概括。
对话历史：
{history_text}
教师总结：
"""
    return call_gpt_segmented(prompt)


def generate_full_dialogue(topic_text, student_id, min_turns=3):
    print(f"\n Generating conversation for {student_id}...")
    history = []
    question = generate_initial_student_question(topic_text)
    print(f"Student (initial question): {question}")
    history.append({"role": "学生", "content": question})
    turns = 0
    label_identity = ""
    label_scenario = ""
    while True:
        turns += 1
        print(f"Round {turns}")
        history_text = "\n".join([f"{h['role']}：{h['content']}" for h in history])
        if turns > 5:
            summary = generate_summary_from_history(history)
            print(f"Teacher: {summary}")
            history.append({"role": "教师", "content": summary})
            break
        teacher_reply = generate_teacher_response(history_text)
        print(f"Teacher: {teacher_reply}")
        history.append({"role": "教师", "content": teacher_reply})
        if "[结束]" in teacher_reply and turns >= min_turns:
            break
        student_reply, identity, scenario = generate_student_response(history)
        print(f"Student: {student_reply}")
        history.append({"role": "学生", "content": student_reply})
        if not label_identity:
            label_identity = identity
        if not label_scenario:
            label_scenario = scenario
    return {
        "student_id": student_id,
        "student_type": label_identity,
        "scenario": label_scenario,
        "dialogue": history,
    }


def load_topics_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_outputs(output_path):
    existing = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    key = f"{item['topic_id']}_{item['student_id']}_{item['repeat_id']}"
                    existing.add(key)
                except Exception:
                    continue
    return existing


def append_dialogue(dialogue_entry, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dialogue_entry, ensure_ascii=False) + "\n")


def generate_single_topic_dialogues(args):
    topic_entry, topic_idx, base_output_dir = args
    topic_text = topic_entry["topic"]
    topic_id = topic_entry.get("id", f"topic_{topic_idx + 1}")
    output_file = os.path.join(base_output_dir, f"multi_dialogue_topic_{topic_id}.json")
    topic_dialogues = []
    for student_idx in range(20):
        student_id = f"Student_{student_idx + 1}"
        for repeat in range(2):
            repeat_id = f"R{repeat + 1}"
            try:
                dialogue = generate_full_dialogue(
                    topic_text, student_id=student_id, min_turns=5
                )
                dialogue["topic_id"] = topic_id
                dialogue["topic_text"] = topic_text
                dialogue["repeat_id"] = repeat_id
                topic_dialogues.append(dialogue)
                time.sleep(1)
            except Exception as e:
                print(f"Dialog generation failed: {e}")
                continue
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(topic_dialogues, f, ensure_ascii=False, indent=2)
    return topic_id


if __name__ == "__main__":
    input_file = os.getenv("MULTI_DIALOGUE_INPUT_FILE", "").strip()
    base_output_dir = os.getenv("OUTPUT_DIR", "").strip()
    if not input_file:
        raise RuntimeError(
            "Please set MULTI_DIALOGUE_INPUT_FILE (topics json) in environment or edit multi_dialogue.py __main__."
        )
    if not base_output_dir:
        raise RuntimeError(
            "Please set OUTPUT_DIR (output directory) in environment or edit multi_dialogue.py __main__."
        )
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    topic_entries = load_topics_from_json(input_file)
    total = len(topic_entries)
    args_list = [(topic_entries[i], i, base_output_dir) for i in range(total)]
    num_processes = min(10, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_single_topic_dialogues, args_list)
    print(
        f"\n All topics have been processed, and a total of {len(results)} topic files have been generated."
    )
