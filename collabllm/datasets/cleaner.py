import argparse
from collections import Counter
import json
import os
import logging
from tqdm import tqdm
import re
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List, Optional, Set

from collabllm.datasets.types import TeachingSession
from collabllm.utils.clean import (
    canonicalize_teaching_intent,
    canonicalize_teaching_strategy,
    normalize_cognitive_level,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DataCleaner: 数据清洗工具，支持规则过滤 + 质量评分 + 多样性平衡抽样
# ============================================================================
# 核心特性：
#   1. 格式检查：对话长度、角色、内容完整性
#   2. 内容质量检查：防止泄题、防止讲课模式、强制提问（苏格拉底法）
#   3. 质量评分：基于标注数据的跨学科、引导等级、认知层级等维度评分
#   4. 多样性平衡抽样：
#      - 按 topic_id 分组，保证所有学科都有代表性
#      - 每个 topic 内按质量排序，取最优样本
#      - 样本在各 topic 之间相对均衡分配
#      - 如果数据不足，从全局高质量池中补齐
# ============================================================================


class DataCleaner:
    def __init__(self, input_path: str, output_path: str, top_n: Optional[int] = None):
        """
        Initialize the DataCleaner with input and output paths.
        Args:
            input_path (str): Path to the input dataset.
            output_path (str): Path to save the cleaned dataset.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.stats = Counter()
        self.dropped_samples = []
        self.valid_sessions: List[TeachingSession] = []
        self.top_n = top_n

    def load_data(self) -> Generator:
        """Yield parsed JSON objects from input path.

        Behavior:
        - If ``self.input_path`` is a file, it will be read directly.
        - If it's a directory, files are discovered recursively (os.walk).
        - Only files ending with ``.jsonl`` or ``.json`` are processed.
        - Skips hidden files and empty lines.
        - Malformed JSON lines are counted in ``self.stats['json_errors']`` and recorded in ``self.dropped_samples``.

        Yields:
            dict: parsed JSON objects (one per line in jsonl/json files).
        """
        if not os.path.exists(self.input_path):
            logger.error("input_path does not exist: %s", self.input_path)
            return

        # Collect files to process (support single file or recursive dir walk)
        if os.path.isfile(self.input_path):
            files: list[str] = [self.input_path]
        else:
            files = []
            for root, _, filenames in os.walk(self.input_path):
                for filename in filenames:
                    if filename.startswith("."):
                        continue
                    if not filename.lower().endswith((".jsonl", ".json")):
                        continue
                    files.append(os.path.join(root, filename))

        for filepath in sorted(files):
            yield from self._read_file(filepath)

    def _read_file(self, path: str) -> Generator:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError as e:
                        self.stats["json_errors"] += 1
                        self.dropped_samples.append({
                            "file": path,
                            "line_no": lineno,
                            "line": raw[:500],
                        })
                        logger.debug("JSON parse failed %s:%d: %s", path, lineno, e)
        except Exception as e:
            logger.exception("Failed to read %s: %s", path, e)

    def filter_format(self, session: TeachingSession) -> tuple[bool, str]:
        """检查基础结构"""
        if len(session.dialogue) < 3:
            return False, "too_short_turns"

        # 检查角色
        valid_roles = {
            "学生",
            "教师",
            # "user",
            # "assistant",
            # "system",
            "teacher",
            "student",
        }
        last_role = None

        for turn in session.dialogue:
            role_lower = turn.role.lower()

            if role_lower not in valid_roles:
                return False, f"invalid_role_{turn.role}"

            if not turn.content.strip():
                return False, "empty_content"

            if role_lower != "system":
                if last_role and role_lower == last_role:
                    return False, "role_repetition"
                last_role = role_lower

        return True, "pass"

    def filter_safety(self, session: TeachingSession) -> tuple[bool, str]:
        """过滤模型拒答、AI身份暴露等有害数据"""
        # 常见拒答关键词
        refusal_patterns = [
            r"作为(一|名|个).*(AI|模型|程序)",
            r"As an AI language model",
            r"我(无法|不能)回答",
            r"违反.*(政策|规定)",
            r"I cannot answer",
            r"涉及.*(敏感|政治)",
            r"截止日期是",  # 幻觉类，比如 knowledge cutoff
        ]

        for turn in session.dialogue:
            if turn.role.lower() in ["教师", "assistant", "teacher"]:
                for pat in refusal_patterns:
                    if re.search(pat, turn.content, re.IGNORECASE):
                        return False, "model_refusal_or_identity"
        return True, "pass"

    def filter_content_quality(self, session: TeachingSession) -> tuple[bool, str]:
        """
        检查内容质量：防止泄题、防止讲课模式、强制提问
        """
        # 预编译正则，提高速度
        # 1. 显式泄露：直接给出定义、公式、结果
        leaked_patterns = [
            r"答案(是|为)",
            r"结果(是|为|等于)",
            # r"定义(是|为)",
            # r"公式(是|为|如下)",
            r"综上所述",
            # r"也就是说",
            r"正确选项",
            r"选[A-D]",
            # r"[\u4e00-\u9fa5]+叫做[\u4e00-\u9fa5]+",  # xx叫做xx
            # r"=\s*\d+",  # 直接写等号数字
        ]

        for i, turn in enumerate(session.dialogue):
            role_lower = turn.role.lower()
            content = turn.content.strip()

            # --- 针对老师的严格审查 ---
            if role_lower in ["教师", "teacher"]:
                if len(content) < 4:
                    return False, "teacher_too_short"

                # 讲课模式检测：如果老师说的话太长（超过 800 字），
                if len(content) > 800:
                    return (
                        False,
                        "teacher_lecture_mode_too_long",
                    )

                # 隐式泄题检测 (Regex)
                for pattern in leaked_patterns:
                    if re.search(pattern, content):
                        # 特殊豁免：如果是最后一轮的总结，允许出现“综上所述”
                        is_last_turn = i == len(session.dialogue) - 1
                        if "综上所述" in pattern and is_last_turn:
                            continue
                        return False, "leaked_answer_pattern"

                # # 苏格拉底核心检测：必须包含问号！
                # # 除非是最后一轮总结，否则老师必须提问（包含 ？ 或 ?）
                # # 这是最狠的一招，能杀掉 80% 的伪苏格拉底数据
                # is_last_turn = i == len(session.dialogue) - 1
                # if not is_last_turn:
                #     if not re.search(r"[？\?]", content):
                #         # 有些老师用“请解释...”代替提问，稍微放宽一点
                #         if not re.search(r"(请|试着|能否|可否|思考|分析)", content):
                #             return False, "not_socratic_no_question"

            # --- 针对学生的检查 (防止数据本身质量低) ---
            elif role_lower in ["学生", "student"]:
                # 如果学生说的话太长（超过 800 字），可能是 GPT 生成的假数据
                if len(content) > 800:
                    return False, "student_too_verbose"

        return True, "pass"

    def calculate_score(self, session: TeachingSession) -> float:
        """
        基于 SID 论文 Table 8 的加权公式计算对话质量分数。
        TotalScore = 0.15*SD + 0.10*SV + 0.15*IKT + 0.15*BP + 0.15*SC + 0.10*L3GR + 0.20*3C
        """
        if not session.annotations:
            return 0.0

        # --- 1. 基础数据统计 ---
        teacher_turns = 0
        strategies_used: List[str] = []
        unique_strategies: Set[str] = set()
        l3_count = 0
        transfer_count = 0
        intents_covered: Set[str] = set()

        student_bloom_levels: List[int] = []
        errors_identified = 0
        errors_corrected = 0

        # 用于 3C (Cognitive Correction) 计算的状态追踪
        last_student_state_was_error = False

        for ann in session.annotations:
            # Annotation 是一个 dataclass，直接使用 getattr 访问属性
            role = getattr(ann, "speaker", "")

            # --- 教师维度统计 ---
            if role in ["教师", "Teacher", "assistant"]:
                teacher_turns += 1

                # 策略统计：使用规范化函数
                strat = getattr(ann, "teaching_strategy", "")
                if strat:
                    strategies_used.append(strat)
                    # 处理可能的逗号分隔 (e.g., "类比, 提示")
                    for s in strat.replace("，", ",").split(","):
                        s_clean = s.strip()
                        if s_clean:
                            # 规范化策略到标准 8 类之一
                            canonical = canonicalize_teaching_strategy(s_clean)
                            if canonical:
                                unique_strategies.add(canonical)

                # L3 引导 [cite: 1338]
                guidance = getattr(ann, "teacher_guidance_level", "")
                if "L3" in guidance:
                    l3_count += 1

                # 跨学科迁移 (IKT) [cite: 1148]
                transfer = getattr(ann, "discipline_transfer", "")
                if transfer in ["是", "Yes", True]:
                    transfer_count += 1

                # 意图覆盖 (SC)：使用规范化函数
                intent = getattr(ann, "teacher_intent", "")
                if intent:
                    canonical_intent = canonicalize_teaching_intent(intent)
                    if canonical_intent:
                        intents_covered.add(canonical_intent)

            # --- 学生维度统计 ---
            elif role in ["学生", "Student", "user"]:
                # Bloom 层级 (BP) [cite: 1147]：使用规范化函数
                cog_level = getattr(ann, "cognitive_level", "")
                level_score = normalize_cognitive_level(cog_level)
                if level_score > 0:
                    student_bloom_levels.append(level_score)

                # 认知状态 (3C 计算)
                state = getattr(ann, "student_cognition_state", "")
                is_error = any(
                    x in state
                    for x in ["模糊", "错误", "Vague", "Incorrect", "Misconception"]
                )
                is_clear = any(
                    x in state for x in ["清晰", "高阶", "Clear", "Higher-order"]
                )

                if last_student_state_was_error:
                    errors_identified += 1
                    if is_clear:
                        errors_corrected += 1

                last_student_state_was_error = is_error

        if teacher_turns == 0:
            return 0.0

        # --- 2. 指标计算 ---

        # (1) SD: Strategy Density
        # 公式: Number of teacher utterances with strategies / Total teacher utterances
        metric_sd = len(strategies_used) / teacher_turns if teacher_turns else 0
        metric_sd = min(metric_sd, 1.0)  # Cap at 1.0

        # (2) SV: Strategy Variety
        # 公式: Unique strategies used / 8
        metric_sv = len(unique_strategies) / 8.0
        metric_sv = min(metric_sv, 1.0)

        # (3) IKT: Interdisciplinary Knowledge Transfer
        # 公式: Transfer counts / Total teacher turns (Paper says counts, but likely meant rate or normalized)
        # 这里为了归一化，我们假设如果 20% 的轮次包含跨学科就是满分 (根据 Table 2 的平均数据推测)
        metric_ikt = (transfer_count / teacher_turns) / 0.2
        metric_ikt = min(metric_ikt, 1.0)

        # (4) BP: Bloom Progression
        # 公式: (Max - Min) / 5
        if student_bloom_levels:
            metric_bp = (max(student_bloom_levels) - min(student_bloom_levels)) / 5.0
        else:
            metric_bp = 0.0
        metric_bp = max(0.0, min(metric_bp, 1.0))

        # (5) SC: Structure Completeness
        # 公式: Covered intents / 4
        # 现在 intents_covered 包含规范化的 4 大类: introduce, check_understanding, guide_reasoning, summarize_enhance
        covered_count = 0
        if "introduce" in intents_covered:
            covered_count += 1
        if "check_understanding" in intents_covered:
            covered_count += 1
        if "guide_reasoning" in intents_covered:
            covered_count += 1
        if "summarize_enhance" in intents_covered:
            covered_count += 1
        metric_sc = covered_count / 4.0

        # (6) L3 GR: L3 Guidance Rate
        # 公式: L3 counts / Total teacher turns
        metric_l3gr = l3_count / teacher_turns

        # (7) 3C: Cognitive Correction Count (Rate)
        # 公式: Successful correction / Total error count
        if errors_identified > 0:
            metric_3c = errors_corrected / errors_identified
        else:
            # 如果没有错误发生，给予一个基准分 (因为没有错误也是一种顺利)
            # 或者设置为 0，视你的筛选偏好而定。SID 论文中几乎所有模型这一项都很高
            metric_3c = 0.8

        # --- 3. 加权汇总 ---
        # TotalScore = 0.15*SD + 0.10*SV + 0.15*IKT + 0.15*BP + 0.15*SC + 0.10*L3GR + 0.20*3C

        final_score = (
            0.15 * metric_sd
            + 0.10 * metric_sv
            + 0.15 * metric_ikt
            + 0.15 * metric_bp
            + 0.15 * metric_sc
            + 0.10 * metric_l3gr
            + 0.20 * metric_3c
        )

        return round(final_score, 4)

    def _select_balanced_samples(self, target_count: int) -> List[TeachingSession]:
        """
        保持多样性的同时追求质量的算法：
        1. 按 topic_id 分组
        2. 每个 topic 组内按 quality_score 倒序排列
        3. 相对均匀地从各 topic 中取样本，直到达到目标数量 target_count
        4. 优先从高质量 topic 中多取，低质量 topic 中少取

        Args:
            target_count (int): 目标样本总数

        Returns:
            List[TeachingSession]: 平衡的样本列表
        """
        logger.info(
            f"Phase 2: Selecting balanced {target_count} samples (diversity + quality)..."
        )

        # Step 1: 按 topic_id 分组
        from collections import defaultdict

        topic_groups = defaultdict(list)
        for session in self.valid_sessions:
            topic_groups[session.topic_id].append(session)

        logger.info(f"Found {len(topic_groups)} unique topics")

        # Step 2: 每个 topic 内排序
        for topic_id in topic_groups:
            topic_groups[topic_id].sort(key=lambda x: x.quality_score, reverse=True)

        # Step 3: 计算每个 topic 的配额（基于均衡分配）
        num_topics = len(topic_groups)

        # 最小化方案：确保每个 topic 至少取 1 条，剩余容量分配给高质量 topic
        if target_count >= num_topics:
            # 每个 topic 基础配额（最多）
            samples_per_topic = target_count // num_topics
            remainder = target_count % num_topics

            # 为每个 topic 分配配额：前 remainder 个 topic 多取 1 条
            topic_quotas = {}
            for idx, topic_id in enumerate(sorted(topic_groups.keys())):
                quota = samples_per_topic + (1 if idx < remainder else 0)
                topic_quotas[topic_id] = quota
        else:
            # target_count < num_topics: 只能从部分 topic 中各取 1 条
            # 优先选择最高质量的 topic
            topic_quality = {
                tid: max(s.quality_score for s in sessions)
                for tid, sessions in topic_groups.items()
            }
            top_topics = sorted(topic_quality.items(), key=lambda x: x[1], reverse=True)
            top_topic_ids = [tid for tid, _ in top_topics[:target_count]]
            topic_quotas = {tid: 1 for tid in top_topic_ids}

        # Step 4: 从每个 topic 中取出配额内的最优样本
        selected_data = []
        remaining_pool = []

        for topic_id in sorted(topic_groups.keys()):
            sessions = topic_groups[topic_id]
            quota = topic_quotas.get(topic_id, 0)

            if quota > 0:
                selected = sessions[:quota]
                selected_data.extend(selected)

                logger.debug(
                    f"Topic '{topic_id}': selected {len(selected)} / {len(sessions)} "
                    f"(score: {selected[-1].quality_score:.2f} ~ {selected[0].quality_score:.2f})"
                )

                # 剩余加入 pool
                remaining = sessions[quota:]
                remaining_pool.extend(remaining)
            else:
                # 这个 topic 不在配额内
                remaining_pool.extend(sessions)

        # 如果因为 target_count < num_topics 导致不足，从 remaining_pool 中补充最高质量的
        current_count = len(selected_data)
        if current_count < target_count and remaining_pool:
            need_fill = target_count - current_count
            remaining_pool.sort(key=lambda x: x.quality_score, reverse=True)
            filled = remaining_pool[:need_fill]
            selected_data.extend(filled)
            logger.info(
                f"Filled {len(filled)} samples from remaining pool "
                f"(score: {filled[-1].quality_score:.2f} ~ {filled[0].quality_score:.2f})"
            )

        # 统计信息
        logger.info(
            f"Final selection: {len(selected_data)} samples (target: {target_count})"
        )
        if selected_data:
            scores = [s.quality_score for s in selected_data]
            logger.info(
                f"Quality score range: {min(scores):.2f} ~ {max(scores):.2f}, "
                f"avg: {sum(scores) / len(scores):.2f}"
            )

        # 返回最终数据
        return selected_data

    def run_pipeline(self) -> None:
        total = 0

        for raw_dict in tqdm(self.load_data(), desc="Cleaning", dynamic_ncols=True):
            total += 1

            # 1. 核心转换：Dict -> Dataclass
            try:
                session = TeachingSession.from_dict(raw_dict)

                # 2. 规则过滤 (传对象进去)
                passed, reason = self.filter_format(session)
                if not passed:
                    self.stats[reason] += 1
                    continue

                passed, reason = self.filter_content_quality(session)
                if not passed:
                    self.stats[reason] += 1
                    if len(self.dropped_samples) < 3:
                        self.dropped_samples.append({
                            "reason": reason,
                            "content": session.to_dict(),
                        })
                    continue

                passed, reason = self.filter_safety(session)
                if not passed:
                    self.stats[reason] += 1
                    continue

                # 3. 质量评分
                session.quality_score = self.calculate_score(session)
                self.valid_sessions.append(session)

            except Exception as e:
                self.stats["schema_error"] += 1
                logger.error("Schema conversion failed: %s", e)
                continue

        logger.info(
            f"Phase 1 Done. Cleaned samples: {len(self.valid_sessions)} / {total}"
        )

        # 2. 排序与截断 (分组 + 多样性保证)
        if self.top_n and len(self.valid_sessions) > self.top_n:
            final_data = self._select_balanced_samples(self.top_n)
        else:
            logger.info(
                "Phase 2: Keeping all cleaned samples (Count < Top N or Top N not set)."
            )
            final_data = self.valid_sessions

        self.save_data(final_data)
        self.print_report(total, len(final_data))

    def save_data(self, data):
        logger.info(f"Saving to {self.output_path}...")
        with open(self.output_path, "w", encoding="utf-8") as f:
            if self.output_path.endswith(".jsonl"):
                for item in data:
                    # Convert Dataclass to dict for JSON serialization
                    data_dict = item.to_dict()
                    f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            else:
                # Convert list of Dataclasses to list of dicts
                serializable_data = [item.to_dict() for item in data]
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    def print_report(self, total, final_count):
        print("\n" + "=" * 50)
        print("PIPELINE FINAL REPORT")
        print("=" * 50)
        print(f"Total Raw Inputs:   {total}")
        print(f"Rule Filtered:      {total - len(self.valid_sessions)}")
        print(f"Rank Filtered:      {len(self.valid_sessions) - final_count}")
        print(f"Final Output:       {final_count}")
        print("-" * 50)
        print("Drop Reasons (Rule-based):")
        for r, c in self.stats.most_common():
            print(f"  - {r}: {c}")
        print("-" * 50)
        if self.valid_sessions:
            print("Quality Score Stats (Valid Data):")
            scores = [s.quality_score for s in self.valid_sessions]
            print(f"  - Avg Score: {sum(scores) / len(scores):.2f}")
            print(f"  - Max Score: {max(scores):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data cleaner with quality filtering and balanced sampling."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data (file or directory with .jsonl/.json files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file (.jsonl or .json)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5000,
        help="Target number of high-quality samples to keep. "
        "Uses balanced sampling across topics to maintain diversity. (default: 5000)",
    )
    args = parser.parse_args()

    cleaner = DataCleaner(args.input, args.output, top_n=args.top_n)
    cleaner.run_pipeline()
