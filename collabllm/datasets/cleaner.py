import argparse
from collections import Counter
import json
import os
import logging
from tqdm import tqdm
import re
from typing import Generator, List, Optional, Set

from collabllm.datasets.types import TeachingSession
from collabllm.utils.metrics import calculate_session_metrics
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
                        self.dropped_samples.append(
                            {
                                "file": path,
                                "line_no": lineno,
                                "line": raw[:500],
                            }
                        )
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
                # 如果学生说的话太长（超过 800 字）
                if len(content) > 800:
                    return False, "student_too_verbose"

        return True, "pass"

    def evaluate_session(self, session: TeachingSession) -> dict:
        """
        Calculates full metrics for the session using the shared metrics utility.
        Assists in both scoring and detailed filtering checks.
        """
        if not session.annotations:
            return {"total_score": 0.0}

        # Ensure turn-level scores are assigned (if used for downstream RL/DPO)
        self.assign_turn_scores(session)

        # Calculate session-level metrics
        return calculate_session_metrics(session)

    def _select_top_percentile(
        self, sessions: List[TeachingSession], percentile: float = 0.35
    ) -> List[TeachingSession]:
        """
        Select the top `percentile` fraction of sessions based on quality_score.
        """
        if not sessions:
            return []

        # Sort by score descending
        sessions.sort(key=lambda s: s.quality_score, reverse=True)

        count = int(len(sessions) * percentile)
        # Ensure at least some minimum if data exists
        count = max(count, min(len(sessions), 100))

        logger.info(
            f"Selecting top {percentile * 100:.1f}%: Keeping {count} / {len(sessions)} samples."
        )
        cutoff_score = sessions[count - 1].quality_score
        logger.info(f"Cutoff Score: {cutoff_score:.4f}")

        return sessions[:count]

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
        metrics_stats = {
            "low_ikt": 0,
            "low_variety": 0,
        }

        for raw_dict in tqdm(self.load_data(), desc="Cleaning", dynamic_ncols=True):
            total += 1

            # 1. Core Conversion: Dict -> Dataclass
            try:
                session = TeachingSession.from_dict(raw_dict)

                # 2. Basic Rule Filtering
                passed, reason = self.filter_format(session)
                if not passed:
                    self.stats[reason] += 1
                    continue

                passed, reason = self.filter_content_quality(session)
                if not passed:
                    self.stats[reason] += 1
                    if len(self.dropped_samples) < 3:
                        self.dropped_samples.append(
                            {
                                "reason": reason,
                                "content": session.to_dict(),
                            }
                        )
                    continue

                passed, reason = self.filter_safety(session)
                if not passed:
                    self.stats[reason] += 1
                    continue

                # 2.5 Clean basic annotations
                removed = session.clean_empty_annotations()
                if not session.annotations:
                    self.stats["empty_annotations_after_cleaning"] += 1
                    continue

                # 3. Advanced Quality Scoring & Filtering
                # Calculate full metrics
                metrics = self.evaluate_session(session)
                session.quality_score = metrics["total_score"]

                # --- Hard Filter 1: Interdisciplinary Knowledge Transfer (IKT) > 0 ---
                if metrics.get("ikt_score", 0) <= 0:
                    self.stats["low_ikt"] += 1
                    metrics_stats["low_ikt"] += 1
                    continue

                # --- Hard Filter 2: Strategy Variety >= 0.5 ---
                if metrics.get("strategy_variety", 0) < 0.5:
                    self.stats["low_strategy_variety"] += 1
                    metrics_stats["low_variety"] += 1
                    continue

                # Passed all hard filters
                self.valid_sessions.append(session)

            except Exception as e:
                self.stats["schema_error"] += 1
                logger.error("Schema conversion failed: %s", e)
                continue

        logger.info(
            f"Phase 1 Done. Cleaned samples: {len(self.valid_sessions)} / {total}"
        )
        logger.info(
            f"Advanced Constraints Drops: IKT={metrics_stats['low_ikt']}, Variety={metrics_stats['low_variety']}"
        )

        # 4. Soft Filter: Top Percentile & Diversity Balance
        # User Strategy: "Cherry-Picking" - Top 30%-40%

        # Sort valid sessions by score descending first
        self.valid_sessions.sort(key=lambda s: s.quality_score, reverse=True)

        if self.top_n:
            logger.info(
                f"Selecting Top {self.top_n} from {len(self.valid_sessions)} valid samples..."
            )
            if len(self.valid_sessions) > self.top_n:
                # Use balanced sampler if we have enough data and need to cut
                final_data = self._select_balanced_samples(self.top_n)
            else:
                final_data = self.valid_sessions
        else:
            # Fallback to percentile if no explicit N given
            final_data = self._select_top_percentile(
                self.valid_sessions, percentile=0.35
            )

        self.save_data(final_data)
        self.print_report(total, len(final_data))

    def assign_turn_scores(self, session: TeachingSession):
        """
        根据标注信息为每一轮教师回复分配即时分数 (Mimic session level calculate_score).

        基于指标模型权重:
        - SD (0.15): 策略密度 (该轮是否使用策略)
        - SV (0.10): 策略多样性 (该轮涉及多少种独特策略)
        - IKT (0.15): 是否存在跨学科迁移
        - SC (0.15): 教学结构完整性 (该轮涉及多少种教学意图)
        - L3GR (0.10): 是否为 L3 级高阶引导
        - 基准分 (0.35): 代替难以局部度量的 BP(0.15) 和 3C(0.20) 指标，确保通过清洗的教师回复有基础回馈
        """
        if not session.annotations:
            return

        def normalize_text(text: str) -> str:
            if not text:
                return ""
            # 只保留中文字符、字母和数字，用于鲁棒匹配
            return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)

        # 创建一个 归一化内容 -> 标注 的映射
        norm_to_ann = {}
        for ann in session.annotations:
            if ann.utterance:
                norm_key = normalize_text(ann.utterance)
                if norm_key:  # 避免空键干扰
                    norm_to_ann[norm_key] = ann

        for turn in session.dialogue:
            if turn.is_teacher():
                # 尝试找到对应的标注
                norm_key = normalize_text(turn.content)
                ann = norm_to_ann.get(norm_key)

                if ann:
                    # 1. 提取策略指标 (SD & SV)
                    strat = getattr(ann, "teaching_strategy", "")
                    unique_strats = set()
                    if strat:
                        for s in strat.replace("，", ",").split(","):
                            s_clean = s.strip()
                            if s_clean:
                                canonical = canonicalize_teaching_strategy(s_clean)
                                if canonical:
                                    unique_strats.add(canonical)

                    metric_sd = 1.0 if unique_strats else 0.0
                    metric_sv = min(len(unique_strats) / 8.0, 1.0)

                    # 2. 跨学科指标 (IKT)
                    transfer = getattr(ann, "discipline_transfer", "")
                    metric_ikt = 1.0 if transfer in ["是", "Yes", True] else 0.0

                    # 3. 意图指标 (SC)
                    intent = getattr(ann, "teacher_intent", "")
                    intents_covered = set()
                    if intent:
                        for i_raw in intent.replace("，", ",").split(","):
                            i_clean = i_raw.strip()
                            if i_clean:
                                canonical_intent = canonicalize_teaching_intent(i_clean)
                                if canonical_intent:
                                    intents_covered.add(canonical_intent)
                    metric_sc = min(len(intents_covered) / 4.0, 1.0)

                    # 4. 引导等级 (L3GR)
                    guidance = getattr(ann, "teacher_guidance_level", "")
                    metric_l3gr = 1.0 if "L3" in guidance else 0.0

                    # 5. 加权汇总
                    # 权重分配参考 calculate_score: 0.15, 0.10, 0.15, 0.15, 0.10
                    weighted_sum = (
                        0.15 * metric_sd
                        + 0.10 * metric_sv
                        + 0.15 * metric_ikt
                        + 0.15 * metric_sc
                        + 0.10 * metric_l3gr
                    )

                    # 最终得分为基准分 (0.35) + 加权分 (max 0.65)
                    turn.score = round(0.35 + weighted_sum, 4)
                else:
                    # 如果老师的话没被标注，但通过了清洗，给一个保底分或 None
                    # 这里选择给 None，表示没有明确的证据评分，聚合时会跳过
                    turn.score = None
            else:
                # 学生回复不赋予教师即时分数
                turn.score = None

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
