import json
import multiprocessing
import os
import random
import time
import logging
from pathlib import Path
import dotenv
import torch

# === 新增：引入进度条库 ===
from tqdm import tqdm

# ===== 0. 全局配置与日志优化 =====
dotenv.load_dotenv()

import warnings

warnings.filterwarnings("ignore")

# 配置日志
# 注意：为了防止日志打断进度条，我们将 StreamHandler (控制台输出) 移除，
# 只保留 FileHandler (文件输出)。控制台进度由 tqdm 独占。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.StreamHandler(),  <-- 注释掉这行，让控制台清爽一点
        logging.FileHandler("generation.log", encoding="utf-8")
    ],
)

# 路径配置
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs").strip()
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ===== Unsloth 本地推理配置 =====
STUDENT_LOCAL_MODEL = os.getenv(
    "STUDENT_LOCAL_MODEL", "unsloth/Qwen3-14B-unsloth-bnb-4bit"
).strip()
TEACHER_BASE_MODEL = os.getenv(
    "TEACHER_BASE_MODEL", "outputs/qwen14b_warmup_merged_4bit"
).strip()
TEACHER_ADAPTER_MODEL = os.getenv(
    "TEACHER_ADAPTER_MODEL", "outputs/dpo_model_1k_3can_opt"
).strip()

MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ_LENGTH", "4096"))
LOAD_IN_4BIT = os.getenv("UNSLOTH_LOAD_IN_4BIT", "1").strip() in {
    "1",
    "true",
    "True",
    "yes",
}
LOCAL_FILES_ONLY = os.getenv("UNSLOTH_LOCAL_FILES_ONLY", "0").strip() in {
    "1",
    "true",
    "True",
    "yes",
}

_MODEL_CACHE = {}


# ===== 1. 核心 LLM 调用函数（Unsloth 本地版） =====


def _load_unsloth_chat_model(model_name: str, adapter_path: str | None = None):
    """加载 Unsloth 模型，可选挂载 LoRA 适配器。"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logging.info(f"[MODEL] Starting to load: {model_name}")
    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": None,
        "load_in_4bit": LOAD_IN_4BIT,
    }

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            **load_kwargs,
            local_files_only=LOCAL_FILES_ONLY,
        )
    except TypeError:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    logging.info(f"[MODEL] Applying chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    if adapter_path:
        logging.info(f"[MODEL] Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    logging.info(f"[MODEL] Switching to inference mode...")
    FastLanguageModel.for_inference(model)
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"[MODEL] Loaded to device: {device}")
    return {"model": model, "tokenizer": tokenizer, "device": device}


def _get_model(role: str):
    if role in _MODEL_CACHE:
        return _MODEL_CACHE[role]

    if role == "teacher":
        logging.info(
            f"Loading teacher model: base={TEACHER_BASE_MODEL}, adapter={TEACHER_ADAPTER_MODEL}"
        )
        _MODEL_CACHE[role] = _load_unsloth_chat_model(
            model_name=TEACHER_BASE_MODEL,
            adapter_path=TEACHER_ADAPTER_MODEL,
        )
        return _MODEL_CACHE[role]

    if role == "student":
        logging.info(f"Loading student model: {STUDENT_LOCAL_MODEL}")
        _MODEL_CACHE[role] = _load_unsloth_chat_model(model_name=STUDENT_LOCAL_MODEL)
        return _MODEL_CACHE[role]

    raise ValueError(f"Unknown role: {role}")


def call_llm(messages, role, temperature=0.7, max_new_tokens=1024):
    client = _get_model(role)
    model = client["model"]
    tokenizer = client["tokenizer"]
    device = client["device"]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logging.debug(f"[GEN] Prompt length (chars): {len(prompt)}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        logging.debug(f"[GEN] Input tokens: {input_len}")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature > 1e-6
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": max(temperature, 1e-4),
                    "top_p": 0.95,
                }
            )

        logging.info(f"[GEN] Starting generation for role={role}, max_new_tokens={max_new_tokens}...")
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        logging.info(f"[GEN] Generation completed.")

        new_tokens = output[0][input_len:]
        raw_content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return raw_content
    except Exception as e:
        logging.error(f"Unsloth generation error ({role}): {str(e)}")
        return None


def build_messages_from_history(history, speaker):
    """将历史对话转换为标准多轮 chat messages 结构。"""
    if speaker == "teacher":
        role_map = {"学生": "user", "教师": "assistant"}
    elif speaker == "student":
        role_map = {"教师": "user", "学生": "assistant"}
    else:
        raise ValueError(f"Unsupported speaker: {speaker}")

    messages = []
    for turn in history:
        msg_role = role_map.get(turn.get("role"))
        content = (turn.get("content") or "").strip()
        if msg_role and content:
            messages.append({"role": msg_role, "content": content})
    return messages


# ===== 2. 业务逻辑 (Prompt) =====


def generate_initial_student_question(topic_text):
    prompt = f"""
你是一名中学生，正在学习一节跨学科课程。请根据下面的课程内容，提出一个你真正感到困惑或好奇的问题。
要求：
1. 回答要口语化，符合中学生身份。
2. 只要输出回答内容，不要输出角色前缀。
3. 不要输出你选择的状态说明（如“（1）学生...”），直接输出对应的对话内容即可。

课程内容如下：
{topic_text}
"""
    return call_llm(
        [{"role": "user", "content": prompt}],
        role="student",
        temperature=1.0,
        max_new_tokens=256,
    )


def generate_teacher_response(history):
    system_prompt = """
你是一位跨学科的教师，始终使用苏格拉底式提问法来引导学生。
目标：通过逐轮递进的问题，引导学生独立思考。

规范：
1. 每一轮只能提出**一个**简洁的问题。
2. 基于学生的回答提炼关键点，推进思考。
3. 鼓励跨学科思考（生物、地理、物理、历史等）。
4. 不要直接给答案，除非学生完全卡住需要提供脚手架。
5. 当判断教学目标达成时，请输出一句简短总结，并**严格以字符串 [结束] 结尾**。
"""
    messages = [{"role": "system", "content": system_prompt.strip()}]
    messages.extend(build_messages_from_history(history, speaker="teacher"))
    messages.append(
        {
            "role": "user",
            "content": "请基于以上对话继续教学。只输出教师当前这一轮的一条回复。",
        }
    )

    return call_llm(
        messages,
        role="teacher",
        temperature=0.9,
        max_new_tokens=512,
    )


def generate_student_response(history):
    student_types = {
        "全优型": "你是一位全优型学生，逻辑清晰，善于跨学科推理。",
        "知识掌握不足": "你是一位基础薄弱的学生，对概念掌握不牢，经常需要老师解释基础词汇。",
        "学习渴望低": "你是一位兴趣不高的学生，回答简短，偶尔会表现出不耐烦。",
    }
    scenarios = [
        "（1）学生不理解问题的含义",
        "（2）学生尝试回答但有部分错误",
        "（3）学生进行了一个类比猜测",
        "（4）学生完全不知道，请求提示",
        "（5）学生回答正确并尝试延伸",
    ]

    identity, instruction = random.choice(list(student_types.items()))
    scenario = random.choice(scenarios)

    system_prompt = f"""
你是一名中学生。请根据教师的问题给出回答。
你的设定：{identity}
当前状态：{scenario}

要求：
1. 回答要口语化，符合中学生身份。
2. 只要输出回答内容，不要输出角色前缀。
"""

    messages = [{"role": "system", "content": system_prompt.strip()}]
    messages.extend(build_messages_from_history(history, speaker="student"))
    messages.append(
        {
            "role": "user",
            "content": "请基于以上对话，只输出学生当前这一轮的回答。",
        }
    )

    reply = call_llm(
        messages,
        role="student",
        temperature=1.0,
        max_new_tokens=256,
    )
    return reply, identity, scenario


def generate_summary(history):
    system_prompt = "请以教师口吻进行简短教学总结，并回答学生最初的问题。"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(build_messages_from_history(history, speaker="teacher"))
    messages.append(
        {
            "role": "user",
            "content": "请基于以上对话给出最终总结与回答。只输出总结内容。",
        }
    )

    return call_llm(
        messages,
        role="teacher",
        temperature=0.7,
        max_new_tokens=512,
    )


def generate_full_dialogue(topic_text, student_id, min_turns=3, max_turns=8):
    history = []

    # 1. 学生提问
    question = generate_initial_student_question(topic_text)
    if not question or "[RateLimit]" in question:
        return None

    history.append({"role": "学生", "content": question})
    logging.info(f"[{student_id}] Question: {question[:30]}...")

    turns = 0
    final_identity = "Mixed"
    final_scenario = "Mixed"

    while turns < max_turns:
        turns += 1

        # 2. 老师回复
        teacher_reply = generate_teacher_response(history)
        if not teacher_reply or "[RateLimit]" in teacher_reply:
            break

        history.append({"role": "教师", "content": teacher_reply})

        if "[结束]" in teacher_reply:
            history[-1]["content"] = teacher_reply.replace("[结束]", "").strip()
            break

        if turns >= max_turns:
            summary = generate_summary(history)
            if summary:
                history.append({"role": "教师", "content": summary})
            break

        # 3. 学生回复
        student_reply, identity, scenario = generate_student_response(history)
        if not student_reply or "[RateLimit]" in student_reply:
            break

        final_identity = identity
        final_scenario = scenario

        history.append({"role": "学生", "content": student_reply})

    return {
        "student_id": student_id,
        "student_type": final_identity,
        "scenario": final_scenario,
        "dialogue": history,
    }


# ===== 3. 多进程与文件写入 =====


def worker_process_topic(args):
    """
    工作进程：处理单个 Topic，生成多个对话
    """
    topic_entry, topic_idx, base_output_dir = args
    topic_text = topic_entry.get("topic", "")
    topic_id = topic_entry.get("id", f"topic_{topic_idx + 1}")

    output_file = os.path.join(base_output_dir, f"dialogue_{topic_id}.jsonl")

    logging.info(f"Start processing Topic: {topic_id}")

    NUM_STUDENTS = 5
    REPEATS = 2

    count = 0
    for s_idx in range(NUM_STUDENTS):
        student_id = f"Student_{s_idx + 1}"
        for r_idx in range(REPEATS):
            try:
                dialogue_data = generate_full_dialogue(topic_text, student_id)

                if dialogue_data:
                    dialogue_data["topic_id"] = topic_id
                    dialogue_data["topic_text_preview"] = topic_text[:50]
                    dialogue_data["repeat_id"] = f"R{r_idx + 1}"

                    # === 实时写入 JSONL ===
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dialogue_data, ensure_ascii=False) + "\n")

                    count += 1
            except Exception as e:
                logging.error(f"Critical Error in worker {topic_id}: {e}")
                time.sleep(1)

    logging.info(f"Finished Topic: {topic_id}, Generated {count} dialogues.")
    return topic_id


def load_topics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    input_file = os.getenv("MULTI_DIALOGUE_INPUT_FILE", "").strip()
    if not input_file or not os.path.exists(input_file):
        print("Error: MULTI_DIALOGUE_INPUT_FILE env var not set or file not found.")
        exit(1)

    topics = load_topics(input_file)

    # Unsloth 本地推理默认单进程，避免多进程重复加载大模型导致显存爆炸
    max_workers = int(os.getenv("MULTI_DIALOGUE_WORKERS", "1"))
    if max_workers < 1:
        max_workers = 1

    print(
        f"Start generating with {max_workers} process(es) using Unsloth local inference..."
    )

    # === 新增：预热一次生成，避免首次 CUDA 编译卡住进度条 ===
    if max_workers == 1:
        logging.info("[WARMUP] Running a tiny warmup generation to initialize CUDA kernels...")
        try:
            _ = call_llm(
                [{"role": "user", "content": "你好"}],
                role="student",
                temperature=1.0,
                max_new_tokens=10,
            )
            logging.info("[WARMUP] Warmup completed.")
        except Exception as e:
            logging.warning(f"[WARMUP] Warmup failed (non-fatal): {e}")

    # 准备参数列表
    args_list = [(t, i, OUTPUT_DIR) for i, t in enumerate(topics)]

    if max_workers == 1:
        for args in tqdm(args_list, total=len(topics), desc="Generating Dialogues", unit="topic"):
            worker_process_topic(args)
    else:
        logging.warning(
            "MULTI_DIALOGUE_WORKERS > 1 may duplicate model loading on each process and cause OOM."
        )
        with multiprocessing.Pool(processes=max_workers) as pool:
            results_iterator = pool.imap_unordered(worker_process_topic, args_list)
            for _ in tqdm(
                results_iterator,
                total=len(topics),
                desc="Generating Dialogues",
                unit="topic",
            ):
                pass

    print("\nAll tasks finished.")
