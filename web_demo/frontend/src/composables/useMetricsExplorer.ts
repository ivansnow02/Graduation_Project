import { computed, ref } from "vue";
import { strFromU8, unzipSync } from "fflate";
import {
  calculateObjectiveMetrics,
  type MetricEntry,
  type ObjectiveAnnotation,
} from "./useEvaluation";

export type { MetricEntry } from "./useEvaluation";

export const METRIC_KEYS = [
  "StrategyDensity",
  "StrategyVariety",
  "IKT",
  "BP",
  "StructureCompleteness",
  "L3GuidanceRate",
  "CognitiveCorrectionRate",
  "TotalScore",
] as const;

export type MetricKey = (typeof METRIC_KEYS)[number];
export type MetricMap = Partial<Record<MetricKey, MetricEntry>>;

export const METRIC_LABELS: Record<MetricKey, string> = {
  StrategyDensity: "策略密度",
  StrategyVariety: "策略多样性",
  IKT: "跨学科迁移",
  BP: "Bloom 进阶",
  StructureCompleteness: "结构完整度",
  L3GuidanceRate: "L3 引导率",
  CognitiveCorrectionRate: "认知纠偏率",
  TotalScore: "总分",
};

export interface MetricSummary {
  modelName: string;
  totalDialogues: number;
  metrics: MetricMap;
  rawValues: Record<string, string>;
  source: "uploaded" | "computed";
}

export interface DialogueTurn {
  role: string;
  content: string;
}

export interface FinalDialogueRecord {
  modelName: string;
  fileKey: string;
  rowIndex: number;
  dialogue: DialogueTurn[];
  annotations: ObjectiveAnnotation[];
  studentId?: string;
  studentType?: string;
  scenario?: string;
  topicId?: string;
  repeatId?: string;
}

export interface MetricDetailRow {
  id: string;
  modelName: string;
  fileKey: string;
  rowIndex: number;
  dialogue?: DialogueTurn[];
  annotations: ObjectiveAnnotation[];
  metrics: MetricMap;
  metricsSource: "uploaded" | "computed" | "missing";
  metricsComplete: boolean;
  studentId?: string;
  studentType?: string;
  scenario?: string;
  topicId?: string;
  repeatId?: string;
}

export interface MetricImportWarning {
  path: string;
  message: string;
}

export interface MetricsImportResult {
  fileName: string;
  importedAt: number;
  summaries: MetricSummary[];
  rows: MetricDetailRow[];
  warnings: MetricImportWarning[];
}

type ZipEntries = Record<string, Uint8Array | string>;

interface UploadedMetricRecord {
  metrics: MetricMap;
  complete: boolean;
}

const IGNORED_PATH_PARTS = new Set(["__MACOSX", ".DS_Store"]);

function normalizePath(path: string) {
  return path.replace(/\\/g, "/").replace(/^\/+/, "");
}

function basename(path: string) {
  const parts = normalizePath(path).split("/");
  return parts[parts.length - 1] ?? path;
}

function stripSuffix(value: string, suffix: string) {
  return value.endsWith(suffix) ? value.slice(0, -suffix.length) : value;
}

function normalizeModelName(value: string) {
  return value.replace(/^dpo(?=\d)/, "dpo_");
}

function shouldIgnorePath(path: string) {
  const normalized = normalizePath(path);
  if (!normalized || normalized.endsWith("/")) return true;
  if (normalized.includes(":Zone.Identifier")) return true;
  return normalized.split("/").some((part) => IGNORED_PATH_PARTS.has(part));
}

function readEntryText(value: Uint8Array | string) {
  return typeof value === "string" ? value : strFromU8(value);
}

function parseScoreFromDisplay(display: string) {
  const trimmed = display.trim();
  if (!trimmed) return 0;
  if (trimmed.endsWith("%")) {
    const parsed = Number.parseFloat(trimmed.slice(0, -1));
    return Number.isFinite(parsed) ? parsed / 100 : 0;
  }
  const parsed = Number.parseFloat(trimmed);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatDisplay(key: MetricKey, score: number) {
  if (key === "TotalScore") return score.toFixed(4);
  return `${(score * 100).toFixed(2)}%`;
}

function normalizeMetricEntry(key: MetricKey, value: unknown): MetricEntry | null {
  if (value && typeof value === "object") {
    const input = value as {
      score?: unknown;
      display?: unknown;
      rawValue?: unknown;
      raw_value?: unknown;
    };
    const display =
      typeof input.display === "string" ? input.display : undefined;
    const score =
      typeof input.score === "number"
        ? input.score
        : display
          ? parseScoreFromDisplay(display)
          : undefined;
    if (typeof score !== "number" || !Number.isFinite(score)) return null;

    const rawCandidate = input.rawValue ?? input.raw_value;
    return {
      score,
      display: display ?? formatDisplay(key, score),
      rawValue:
        typeof rawCandidate === "number" && Number.isFinite(rawCandidate)
          ? rawCandidate
          : undefined,
    };
  }

  if (typeof value === "string") {
    return {
      score: parseScoreFromDisplay(value),
      display: value,
    };
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return {
      score: value,
      display: formatDisplay(key, value),
    };
  }

  return null;
}

function normalizeMetricMap(value: unknown): MetricMap {
  if (!value || typeof value !== "object") return {};
  const input = value as Record<string, unknown>;
  const metrics: MetricMap = {};
  for (const key of METRIC_KEYS) {
    const entry = normalizeMetricEntry(key, input[key]);
    if (entry) metrics[key] = entry;
  }
  return metrics;
}

function hasCompleteMetrics(metrics: MetricMap) {
  return METRIC_KEYS.every((key) => Boolean(metrics[key]));
}

function parseFinalJsonl(
  path: string,
  modelName: string,
  fileKey: string,
  text: string,
  warnings: MetricImportWarning[],
) {
  const rows: FinalDialogueRecord[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]?.trim();
    if (!line) continue;
    try {
      const parsed = JSON.parse(line) as Record<string, unknown>;
      const dialogue = Array.isArray(parsed.dialogue)
        ? parsed.dialogue
            .filter((turn): turn is DialogueTurn => {
              return (
                Boolean(turn) &&
                typeof turn === "object" &&
                typeof (turn as DialogueTurn).role === "string" &&
                typeof (turn as DialogueTurn).content === "string"
              );
            })
            .map((turn) => ({ role: turn.role, content: turn.content }))
        : [];
      const annotations = Array.isArray(parsed.annotations)
        ? (parsed.annotations.filter(
            (ann) =>
              Boolean(ann) &&
              typeof ann === "object" &&
              typeof (ann as ObjectiveAnnotation).speaker === "string" &&
              typeof (ann as ObjectiveAnnotation).utterance === "string",
          ) as ObjectiveAnnotation[])
        : [];

      rows.push({
        modelName,
        fileKey,
        rowIndex: rows.length,
        dialogue,
        annotations,
        studentId:
          typeof parsed.student_id === "string" ? parsed.student_id : undefined,
        studentType:
          typeof parsed.student_type === "string"
            ? parsed.student_type
            : undefined,
        scenario: typeof parsed.scenario === "string" ? parsed.scenario : undefined,
        topicId: typeof parsed.topic_id === "string" ? parsed.topic_id : undefined,
        repeatId: typeof parsed.repeat_id === "string" ? parsed.repeat_id : undefined,
      });
    } catch (error) {
      warnings.push({
        path,
        message: `第 ${i + 1} 行不是有效 JSON：${String(error)}`,
      });
    }
  }
  return rows;
}

function pushNested<T>(
  target: Map<string, Map<string, T[]>>,
  modelName: string,
  fileKey: string,
  values: T[],
) {
  const modelBucket = target.get(modelName) ?? new Map<string, T[]>();
  modelBucket.set(fileKey, values);
  target.set(modelName, modelBucket);
}

function detectFinalPath(path: string) {
  const parts = normalizePath(path).split("/");
  const finalIndex = parts.lastIndexOf("final");
  if (finalIndex === -1 || !parts[finalIndex + 1]) return null;
  const name = basename(path);
  if (!name.endsWith(".jsonl")) return null;
  return {
    modelName: normalizeModelName(parts[finalIndex + 1]),
    fileKey: stripSuffix(name, ".jsonl"),
  };
}

function detectMetricPath(path: string) {
  const parts = normalizePath(path).split("/");
  const metricsIndex = parts.lastIndexOf("metrics");
  if (metricsIndex === -1) return null;
  const relativeParts = parts.slice(metricsIndex + 1);
  const name = basename(path);
  if (!name.endsWith(".json")) return null;

  if (relativeParts.length === 1) {
    return {
      kind: "summary" as const,
      modelName: normalizeModelName(stripSuffix(name, ".json")),
      fileKey: "",
    };
  }

  const parent = relativeParts[relativeParts.length - 2] ?? "";
  if (parent.endsWith("_per_file_metrics")) {
    return {
      kind: "detail" as const,
      modelName: normalizeModelName(stripSuffix(parent, "_per_file_metrics")),
      fileKey: stripSuffix(stripSuffix(name, ".json"), "_metrics"),
    };
  }

  return null;
}

function parseUploadedMetricRecords(
  path: string,
  text: string,
  warnings: MetricImportWarning[],
) {
  try {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) {
      warnings.push({ path, message: "逐文件指标 JSON 应为数组" });
      return [];
    }
    return parsed.map((item): UploadedMetricRecord => {
      const metrics = normalizeMetricMap(
        item && typeof item === "object"
          ? (item as Record<string, unknown>).metrics_results
          : null,
      );
      return { metrics, complete: hasCompleteMetrics(metrics) };
    });
  } catch (error) {
    warnings.push({ path, message: `指标 JSON 解析失败：${String(error)}` });
    return [];
  }
}

function parseSummary(
  modelName: string,
  path: string,
  text: string,
  warnings: MetricImportWarning[],
) {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    const totalDialogues =
      typeof parsed.total_dialogues_processed === "number"
        ? parsed.total_dialogues_processed
        : 0;
    const rawValues =
      parsed.average_raw_values && typeof parsed.average_raw_values === "object"
        ? Object.fromEntries(
            Object.entries(parsed.average_raw_values as Record<string, unknown>)
              .filter(([, value]) => typeof value === "string")
              .map(([key, value]) => [key, value as string]),
          )
        : {};
    return {
      modelName,
      totalDialogues,
      metrics: normalizeMetricMap(parsed.average_scores),
      rawValues,
      source: "uploaded" as const,
    };
  } catch (error) {
    warnings.push({ path, message: `汇总指标 JSON 解析失败：${String(error)}` });
    return null;
  }
}

function computeSummary(modelName: string, rows: MetricDetailRow[]): MetricSummary {
  const metrics: MetricMap = {};
  const rawValues: Record<string, string> = {};
  for (const key of METRIC_KEYS) {
    const entries = rows
      .map((row) => row.metrics[key])
      .filter((entry): entry is MetricEntry => Boolean(entry));
    if (entries.length === 0) continue;
    const score =
      entries.reduce((sum, entry) => sum + entry.score, 0) / entries.length;
    const rawEntries = entries
      .map((entry) => entry.rawValue)
      .filter((value): value is number => typeof value === "number");
    if (rawEntries.length > 0) {
      const rawAverage =
        rawEntries.reduce((sum, value) => sum + value, 0) / rawEntries.length;
      rawValues[`${key}_raw_value`] = rawAverage.toFixed(2);
    }
    metrics[key] = { score, display: formatDisplay(key, score) };
  }

  return {
    modelName,
    totalDialogues: rows.length,
    metrics,
    rawValues,
    source: "computed",
  };
}

function mergeRows(
  finalRecords: Map<string, Map<string, FinalDialogueRecord[]>>,
  metricRecords: Map<string, Map<string, UploadedMetricRecord[]>>,
  warnings: MetricImportWarning[],
) {
  const rows: MetricDetailRow[] = [];
  const modelNames = new Set([...finalRecords.keys(), ...metricRecords.keys()]);

  for (const modelName of [...modelNames].sort((a, b) => a.localeCompare(b))) {
    const finalByFile = finalRecords.get(modelName) ?? new Map();
    const metricsByFile = metricRecords.get(modelName) ?? new Map();
    const fileKeys = new Set([...finalByFile.keys(), ...metricsByFile.keys()]);

    for (const fileKey of [...fileKeys].sort((a, b) => a.localeCompare(b))) {
      const finalRows = finalByFile.get(fileKey) ?? [];
      const metricRows = metricsByFile.get(fileKey) ?? [];
      if (finalRows.length && metricRows.length && finalRows.length !== metricRows.length) {
        warnings.push({
          path: `${modelName}/${fileKey}`,
          message: `对话 ${finalRows.length} 条，指标 ${metricRows.length} 条，仅按行号匹配重叠部分`,
        });
      }

      const total = Math.max(finalRows.length, metricRows.length);
      for (let rowIndex = 0; rowIndex < total; rowIndex += 1) {
        const finalRow = finalRows[rowIndex];
        const metricRow = metricRows[rowIndex];
        const computedMetrics = finalRow?.annotations.length
          ? calculateObjectiveMetrics(finalRow.annotations)
          : null;
        const metrics = metricRow?.complete
          ? metricRow.metrics
          : computedMetrics ?? metricRow?.metrics ?? {};
        const metricsSource = metricRow?.complete
          ? "uploaded"
          : computedMetrics
            ? "computed"
            : "missing";

        rows.push({
          id: `${modelName}:${fileKey}:${rowIndex}`,
          modelName,
          fileKey,
          rowIndex,
          dialogue: finalRow?.dialogue,
          annotations: finalRow?.annotations ?? [],
          metrics,
          metricsSource,
          metricsComplete: hasCompleteMetrics(metrics),
          studentId: finalRow?.studentId,
          studentType: finalRow?.studentType,
          scenario: finalRow?.scenario,
          topicId: finalRow?.topicId,
          repeatId: finalRow?.repeatId,
        });
      }
    }
  }

  return rows;
}

export function parseMetricsZipEntries(
  entries: ZipEntries,
  fileName = "uploaded.zip",
): MetricsImportResult {
  const warnings: MetricImportWarning[] = [];
  const finalRecords = new Map<string, Map<string, FinalDialogueRecord[]>>();
  const metricRecords = new Map<string, Map<string, UploadedMetricRecord[]>>();
  const summaries = new Map<string, MetricSummary>();

  for (const [rawPath, entry] of Object.entries(entries)) {
    const path = normalizePath(rawPath);
    if (shouldIgnorePath(path)) continue;
    const text = readEntryText(entry);

    const finalInfo = detectFinalPath(path);
    if (finalInfo) {
      pushNested(
        finalRecords,
        finalInfo.modelName,
        finalInfo.fileKey,
        parseFinalJsonl(path, finalInfo.modelName, finalInfo.fileKey, text, warnings),
      );
      continue;
    }

    const metricInfo = detectMetricPath(path);
    if (!metricInfo) continue;
    if (metricInfo.kind === "summary") {
      const summary = parseSummary(metricInfo.modelName, path, text, warnings);
      if (summary) summaries.set(metricInfo.modelName, summary);
      continue;
    }

    pushNested(
      metricRecords,
      metricInfo.modelName,
      metricInfo.fileKey,
      parseUploadedMetricRecords(path, text, warnings),
    );
  }

  const rows = mergeRows(finalRecords, metricRecords, warnings);
  const rowsByModel = new Map<string, MetricDetailRow[]>();
  for (const row of rows) {
    const bucket = rowsByModel.get(row.modelName) ?? [];
    bucket.push(row);
    rowsByModel.set(row.modelName, bucket);
  }
  for (const [modelName, modelRows] of rowsByModel) {
    if (!summaries.has(modelName)) {
      summaries.set(modelName, computeSummary(modelName, modelRows));
    }
  }

  return {
    fileName,
    importedAt: Date.now(),
    summaries: [...summaries.values()].sort((a, b) =>
      a.modelName.localeCompare(b.modelName),
    ),
    rows,
    warnings,
  };
}

export async function parseMetricsZipFile(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return parseMetricsZipEntries(unzipSync(bytes), file.name);
}

export async function parseMetricsZipUrl(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`ZIP 下载失败：HTTP ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const fileName = decodeURIComponent(url.split("/").pop()?.split("?")[0] || "remote.zip");
  return parseMetricsZipEntries(unzipSync(bytes), fileName);
}

export function useMetricsExplorer() {
  const result = ref<MetricsImportResult | null>(null);
  const loading = ref(false);
  const error = ref("");

  const modelOptions = computed(() => result.value?.summaries.map((item) => item.modelName) ?? []);
  const fileOptions = computed(() => {
    const keys = new Set(result.value?.rows.map((row) => row.fileKey) ?? []);
    return [...keys].sort((a, b) => a.localeCompare(b));
  });

  async function importZip(file: File) {
    loading.value = true;
    error.value = "";
    try {
      result.value = await parseMetricsZipFile(file);
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
    } finally {
      loading.value = false;
    }
  }

  async function importZipUrl(url: string) {
    const trimmed = url.trim();
    if (!trimmed) return;
    loading.value = true;
    error.value = "";
    try {
      result.value = await parseMetricsZipUrl(trimmed);
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
    } finally {
      loading.value = false;
    }
  }

  function reset() {
    result.value = null;
    error.value = "";
  }

  return {
    result,
    loading,
    error,
    modelOptions,
    fileOptions,
    importZip,
    importZipUrl,
    reset,
  };
}
