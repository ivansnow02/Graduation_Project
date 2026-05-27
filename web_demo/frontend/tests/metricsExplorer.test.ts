import { describe, expect, test } from "bun:test";
import { parseMetricsZipEntries } from "../src/composables/useMetricsExplorer";

const dialogueRecord = {
  student_id: "Student_1",
  student_type: "全优型",
  scenario: "测试场景",
  topic_id: "topic_1",
  repeat_id: "R1",
  dialogue: [
    { role: "学生", content: "问题一" },
    { role: "教师", content: "引导一" },
  ],
  annotations: [
    {
      speaker: "学生",
      utterance: "问题一",
      teacher_intent: "",
      teaching_strategy: "",
      discipline: "地理",
      discipline_transfer: "否",
      student_cognition_state: "错误回答",
      teacher_guidance_level: "",
      cognitive_level: "理解",
    },
    {
      speaker: "教师",
      utterance: "引导一",
      teacher_intent: "引导推理",
      teaching_strategy: "追问",
      discipline: "地理",
      discipline_transfer: "否",
      student_cognition_state: "",
      teacher_guidance_level: "L3",
      cognitive_level: "分析",
    },
  ],
};

const uploadedMetrics = [
  {
    dialogue_id: null,
    metrics_results: {
      StrategyDensity: { score: 1, display: "100.00%" },
      StrategyVariety: { score: 0.125, display: "12.50%" },
      IKT: { score: 0, display: "0.00%" },
      BP: { raw_value: 1, score: 0.2, display: "20.00%" },
      StructureCompleteness: { score: 0.25, display: "25.00%" },
      L3GuidanceRate: { score: 1, display: "100.00%" },
      CognitiveCorrectionRate: { score: 0, display: "0.00%" },
      TotalScore: { score: 0.3475, display: "0.3475" },
    },
  },
];

describe("parseMetricsZipEntries", () => {
  test("匹配 final JSONL 与逐文件 metrics", () => {
    const result = parseMetricsZipEntries(
      {
        "data/final/baseline/dialogue_topic_1.jsonl": `${JSON.stringify(dialogueRecord)}\n`,
        "data/metrics/baseline.json": JSON.stringify({
          total_dialogues_processed: 1,
          average_scores: {
            StrategyDensity: "100.00%",
            StrategyVariety: "12.50%",
            IKT: "0.00%",
            BP: "20.00%",
            StructureCompleteness: "25.00%",
            L3GuidanceRate: "100.00%",
            CognitiveCorrectionRate: "0.00%",
            TotalScore: "0.3475",
          },
          average_raw_values: { BP_raw_value: "1.00" },
        }),
        "data/metrics/baseline_per_file_metrics/dialogue_topic_1_metrics.json": JSON.stringify(uploadedMetrics),
      },
      "fixture.zip",
    );

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0].modelName).toBe("baseline");
    expect(result.rows[0].fileKey).toBe("dialogue_topic_1");
    expect(result.rows[0].metricsSource).toBe("uploaded");
    expect(result.rows[0].metrics.TotalScore?.display).toBe("0.3475");
    expect(result.summaries[0].totalDialogues).toBe(1);
    expect(result.warnings).toHaveLength(0);
  });

  test("缺失上传指标时从 annotations 重算", () => {
    const result = parseMetricsZipEntries({
      "final/baseline/dialogue_topic_1.jsonl": `${JSON.stringify(dialogueRecord)}\n`,
    });

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0].metricsSource).toBe("computed");
    expect(result.rows[0].metrics.TotalScore?.display).toBeTruthy();
    expect(result.summaries[0].source).toBe("computed");
  });

  test("记录无效 JSONL 和行数不匹配 warning", () => {
    const result = parseMetricsZipEntries({
      "final/baseline/dialogue_topic_1.jsonl": `${JSON.stringify(dialogueRecord)}\n{bad}\n`,
      "metrics/baseline_per_file_metrics/dialogue_topic_1_metrics.json": JSON.stringify([
        ...uploadedMetrics,
        ...uploadedMetrics,
      ]),
    });

    expect(result.rows).toHaveLength(2);
    expect(result.warnings.some((item) => item.message.includes("不是有效 JSON"))).toBe(true);
    expect(result.warnings.some((item) => item.message.includes("仅按行号匹配"))).toBe(true);
  });
});
