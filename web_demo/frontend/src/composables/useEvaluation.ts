import { computed } from "vue";
import { useLLM, type ModelConfig, type ChatMessage } from "./useLLM";

export interface DialogueTurn {
  role: "学生" | "教师";
  content: string;
}

export interface ObjectiveAnnotation {
  speaker: string;
  utterance: string;
  teacher_intent?: string;
  teaching_strategy?: string;
  discipline?: string;
  discipline_transfer?: string;
  student_cognition_state?: string;
  teacher_guidance_level?: string;
  cognitive_level?: string;
}

export type AnnotationFieldKey =
  | "teacher_intent"
  | "teaching_strategy"
  | "discipline"
  | "discipline_transfer"
  | "student_cognition_state"
  | "teacher_guidance_level"
  | "cognitive_level";

export interface MetricEntry {
  score: number;
  display: string;
  rawValue?: number;
}

export interface ObjectiveMetrics {
  StrategyDensity: MetricEntry;
  StrategyVariety: MetricEntry;
  IKT: MetricEntry;
  BP: MetricEntry;
  StructureCompleteness: MetricEntry;
  L3GuidanceRate: MetricEntry;
  CognitiveCorrectionRate: MetricEntry;
  TotalScore: MetricEntry;
}

export interface EvaluationRecord {
  dialogueId: string;
  evaluatedAt: number;
  sourceHash: string;
  modelName: string;
  dialogue: DialogueTurn[];
  annotations: ObjectiveAnnotation[];
  originalAnnotations: ObjectiveAnnotation[];
  metrics: ObjectiveMetrics;
}

export const TEACHER_INTENT_OPTIONS = [
  "引出概念",
  "检测理解",
  "引导推理",
  "引发迁移",
  "总结提升",
] as const;
export const DISCIPLINE_TRANSFER_OPTIONS = ["是", "否"] as const;
export const STUDENT_COGNITION_STATE_OPTIONS = [
  "清晰理解",
  "模糊理解",
  "表达困难",
  "答非所问",
  "错误回答",
  "高阶思考",
] as const;
export const TEACHER_GUIDANCE_LEVEL_OPTIONS = ["L1", "L2", "L3"] as const;
export const COGNITIVE_LEVEL_OPTIONS = [
  "记忆",
  "理解",
  "应用",
  "分析",
  "评价",
  "创造",
] as const;
export const TEACHING_STRATEGY_OPTIONS = [
  "情境设问",
  "追问",
  "类比",
  "提示",
  "拆解问题",
  "鼓励回应",
  "正误反馈",
  "引导推理",
] as const;

const TOTAL_POSSIBLE_STRATEGIES = TEACHING_STRATEGY_OPTIONS.length;
const REQUIRED_INTENTS = new Set([
  "引出概念",
  "引导推理",
  "引发迁移",
  "总结提升",
]);
const BLOOM_MAP: Record<string, number> = {
  记忆: 1,
  理解: 2,
  应用: 3,
  分析: 4,
  评价: 5,
  创造: 6,
};
const MAX_BLOOM_PROGRESSION = Object.keys(BLOOM_MAP).length - 1;

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function stableHash(value: unknown): string {
  const text = JSON.stringify(value);
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash).toString(36);
}

export function cloneAnnotations(
  annotations: ObjectiveAnnotation[],
): ObjectiveAnnotation[] {
  return annotations.map((annotation) => ({ ...annotation }));
}

function splitListValue(value?: string): string[] {
  return (value ?? "")
    .replace(/，/g, ",")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function buildEvaluationDialogue(
  messages: { role: "user"; content: string }[],
  responses: Record<number, string>,
): DialogueTurn[] {
  const dialogue: DialogueTurn[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    const studentText = messages[i]?.content?.trim();
    const teacherText = responses[i]?.trim();
    if (!studentText || !teacherText) continue;
    dialogue.push({ role: "学生", content: studentText });
    dialogue.push({ role: "教师", content: teacherText });
  }
  return dialogue;
}

export function createEvaluationSourceHash(
  messages: { role: "user"; content: string }[],
  responses: Record<number, string>,
): string {
  return stableHash(buildEvaluationDialogue(messages, responses));
}

export function calculateObjectiveMetrics(
  annotations: ObjectiveAnnotation[],
): ObjectiveMetrics | null {
  if (annotations.length === 0) return null;

  const teacherUtterances = annotations.filter((ann) => ann.speaker === "教师");
  const studentUtterances = annotations.filter((ann) => ann.speaker === "学生");
  const teacherUtteranceCount = teacherUtterances.length;
  const strategyUtteranceCount = teacherUtterances.filter(
    (ann) => ann.teaching_strategy,
  ).length;
  const strategyDensity =
    teacherUtteranceCount > 0
      ? strategyUtteranceCount / teacherUtteranceCount
      : 0;

  const usedStrategies = new Set(
    teacherUtterances
      .flatMap((ann) => splitListValue(ann.teaching_strategy))
      .filter((strategy): strategy is string => Boolean(strategy)),
  );
  const strategyVariety =
    TOTAL_POSSIBLE_STRATEGIES > 0
      ? usedStrategies.size / TOTAL_POSSIBLE_STRATEGIES
      : 0;

  const crossDisciplinaryJumps = annotations.filter(
    (ann) => ann.discipline_transfer === "是",
  ).length;
  const finalIKTScore =
    teacherUtteranceCount > 0
      ? crossDisciplinaryJumps / teacherUtteranceCount
      : 0;

  const studentCognitionLevels = studentUtterances
    .map((ann) => BLOOM_MAP[ann.cognitive_level ?? ""])
    .filter((level): level is number => typeof level === "number");

  const bloomProgressionRaw =
    studentCognitionLevels.length > 0
      ? Math.max(...studentCognitionLevels) -
        Math.min(...studentCognitionLevels)
      : 0;
  const normBloomProgression =
    MAX_BLOOM_PROGRESSION > 0 ? bloomProgressionRaw / MAX_BLOOM_PROGRESSION : 0;

  const teacherIntents = new Set(
    teacherUtterances
      .map((ann) => ann.teacher_intent?.trim())
      .filter((intent): intent is string => Boolean(intent)),
  );
  const coveredIntentsCount = [...teacherIntents].filter((intent) =>
    REQUIRED_INTENTS.has(intent),
  ).length;
  const structureCompleteness =
    REQUIRED_INTENTS.size > 0 ? coveredIntentsCount / REQUIRED_INTENTS.size : 0;

  const l3GuidanceCount = teacherUtterances.filter(
    (ann) => ann.teacher_guidance_level === "L3",
  ).length;
  const l3GuidanceRate =
    teacherUtteranceCount > 0 ? l3GuidanceCount / teacherUtteranceCount : 0;

  const studentCognitionStates = studentUtterances.map(
    (ann) => ann.student_cognition_state,
  );
  const totalErrorCount = studentCognitionStates.filter(
    (state) => state === "错误回答",
  ).length;
  let successfulCorrectionCount = 0;
  for (let i = 0; i < studentCognitionStates.length - 1; i += 1) {
    const current = studentCognitionStates[i];
    const next = studentCognitionStates[i + 1];
    if (
      current === "错误回答" &&
      (next === "高阶思考" || next === "清晰理解")
    ) {
      successfulCorrectionCount += 1;
    }
  }
  const cognitiveCorrectionRate =
    totalErrorCount === 0 ? 1 : successfulCorrectionCount / totalErrorCount;

  const totalScore =
    0.15 * strategyDensity +
    0.1 * strategyVariety +
    0.15 * finalIKTScore +
    0.15 * normBloomProgression +
    0.15 * structureCompleteness +
    0.1 * l3GuidanceRate +
    0.2 * cognitiveCorrectionRate;

  return {
    StrategyDensity: {
      score: strategyDensity,
      display: formatPercent(strategyDensity),
    },
    StrategyVariety: {
      score: strategyVariety,
      display: formatPercent(strategyVariety),
    },
    IKT: { score: finalIKTScore, display: formatPercent(finalIKTScore) },
    BP: {
      rawValue: bloomProgressionRaw,
      score: normBloomProgression,
      display: formatPercent(normBloomProgression),
    },
    StructureCompleteness: {
      score: structureCompleteness,
      display: formatPercent(structureCompleteness),
    },
    L3GuidanceRate: {
      score: l3GuidanceRate,
      display: formatPercent(l3GuidanceRate),
    },
    CognitiveCorrectionRate: {
      score: cognitiveCorrectionRate,
      display: formatPercent(cognitiveCorrectionRate),
    },
    TotalScore: { score: totalScore, display: totalScore.toFixed(4) },
  };
}

function buildAnnotationPrompt(dialogue: DialogueTurn[]): string {
  const dialogueText = dialogue
    .map((turn) => `${turn.role}：${turn.content}`)
    .join("\n");
  return `
你是一位教育认知标注专家，请根据下面一段教学对话，逐轮提取以下9项教学信息，并输出为结构化 JSON 格式（列表形式）。
每一轮包含教师或学生的一个发言。请不要跳过任何一轮。

## 【需要标注的字段】：
- speaker：发言者（"教师"或"学生"）
- utterance：原始发言文本，不能进行任何修改
- teacher_intent：教师发言中体现的教学目的，有以下五种："引出概念"、"检测理解"、"引导推理"、"引发迁移"、"总结提升"，学生轮为空字符串
- teaching_strategy：教师采用的策略，请从"情境设问"、"追问"、"类比"、"提示"、"拆解问题"、"鼓励回应"、"正误反馈"、"引导推理"中选择；多个策略用逗号分隔，学生轮为空字符串
- discipline：该轮涉及的学科，如"地理"、"生物"、"物理"、"历史"，多个学科请用逗号分隔
- discipline_transfer：若当前轮相较上轮出现新的学科，引导学科间联系，请填"是"，否则填"否"
- student_cognition_state：仅学生轮填写，有以下几种："清晰理解"、"模糊理解"、"表达困难"、"答非所问"、"错误回答"、"高阶思考"；教师轮为空字符串
- teacher_guidance_level：仅教师轮填写，分为 L1、L2、L3 三种；学生轮为空字符串
- cognitive_level：请根据 Bloom 分类选择以下之一："记忆"、"理解"、"应用"、"分析"、"评价"、"创造"

##【对话内容】：
${dialogueText}

##【输出要求】：
只输出严格 JSON 数组，不要输出代码块或解释。
`.trim();
}

function extractFirstJsonArray(text: string): string | null {
  const start = text.indexOf("[");
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    if (text[i] === "[") depth += 1;
    if (text[i] === "]") {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function parseAnnotations(text: string): ObjectiveAnnotation[] {
  // 从模型返回的文本中提取首个 JSON 数组片段并解析为 annotations
  const jsonText = extractFirstJsonArray(text);
  if (!jsonText) {
    throw new Error("评测模型没有返回 JSON 数组");
  }
  const parsed = JSON.parse(jsonText);
  if (!Array.isArray(parsed)) {
    throw new Error("评测模型返回的 JSON 不是数组");
  }
  // 过滤并验证每条记录包含必须字段：speaker 与 utterance
  return parsed.filter((item): item is ObjectiveAnnotation => {
    return Boolean(
      item && typeof item === "object" && item.speaker && item.utterance,
    );
  });
}

function assertEvaluatorConfig(model: ModelConfig) {
  if (!model.name || !model.apiKey || !model.baseUrl) {
    throw new Error("请先在全局设置中配置评测模型");
  }
}

// `useEvaluation` 负责调用评测模型生成逐轮 annotation，
// 并封装评价计算与可用性检查逻辑
export function useEvaluation() {
  const { completeChat } = useLLM();

  async function evaluateDialogue(
    model: ModelConfig,
    dialogue: DialogueTurn[],
    sourceHash: string,
  ): Promise<EvaluationRecord> {
    assertEvaluatorConfig(model);
    if (dialogue.length < 2) {
      throw new Error("当前会话还没有可评测的完整师生轮次");
    }

    const messages: ChatMessage[] = [
      {
        role: "system",
        content: "你是一位严谨的教育认知标注专家，只输出用户要求的 JSON。",
      },
      { role: "user", content: buildAnnotationPrompt(dialogue) },
    ];
    // 调用评测模型（同步完成），该模型应严格返回一个 JSON 数组
    const rawText = await completeChat(model, messages, {
      temperature: 0,
      maxTokens: 5000,
    });
    // 解析并验证 annotations，若解析失败则抛错交由上层处理
    const annotations = parseAnnotations(rawText);
    const metrics = calculateObjectiveMetrics(annotations);
    if (!metrics) {
      throw new Error("未解析到有效 annotations，无法计算客观指标");
    }

    return {
      dialogueId: `webdemo-${Date.now()}`,
      evaluatedAt: Date.now(),
      sourceHash,
      modelName: model.name,
      dialogue,
      annotations,
      originalAnnotations: cloneAnnotations(annotations),
      metrics,
    };
  }

  function useEvaluationAvailability(options: {
    chatMode: { value: string };
    streaming: { value: boolean };
    sourceHash: { value: string };
  }) {
    return computed(() => {
      return (
        options.chatMode.value === "single" &&
        !options.streaming.value &&
        Boolean(options.sourceHash.value)
      );
    });
  }

  return { evaluateDialogue, useEvaluationAvailability };
}
