<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  METRIC_KEYS,
  METRIC_LABELS,
  isAblationModelName,
  useMetricsExplorer,
  type MetricDetailRow,
  type MetricEntry,
  type MetricKey,
  type MetricSummary,
} from "../composables/useMetricsExplorer";

const {
  result,
  loading,
  error,
  modelOptions,
  fileOptions,
  importZip,
  importZipUrl,
  reset,
} = useMetricsExplorer();

const selectedModel = ref("");
const selectedFile = ref("");
const selectedMetric = ref<MetricKey>("TotalScore");
const chartMetric = ref<MetricKey>("TotalScore");
const selectedChartModels = ref<string[]>([]);
const searchText = ref("");
const selectedRowId = ref("");
const zipUrl = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const activeMetricsTab = ref<"ablation" | "comparison" | "details">("ablation");
const selectedAblationModuleId = ref("seven_dim");
const selectedAblationComparisonId = ref("seven_dim_overview");

interface AblationGroupDesign {
  group: string;
  sftData: string;
  margin: string;
  window: string;
  special: string;
  goal: string;
}

interface AblationModule {
  id: string;
  stageId: string;
  title: string;
  description: string;
  groups: string[];
}

interface AblationComparisonOption {
  id: string;
  label: string;
  description: string;
  groups: string[];
  note: string;
}

interface AblationSection {
  id: string;
  title: string;
  description: string;
  options: AblationComparisonOption[];
}

const ABLATION_DESIGNS: AblationGroupDesign[] = [
  { group: "g1", sftData: "—", margin: "—", window: "—", special: "—", goal: "基线" },
  { group: "g2", sftData: "SID", margin: "—", window: "—", special: "—", goal: "SFT 预热效果" },
  { group: "g3", sftData: "—", margin: "✗", window: "2", special: "—", goal: "无 SFT 直接 DPO" },
  { group: "g4", sftData: "SID", margin: "✗", window: "2", special: "—", goal: "多轮 DPO 无 Margin" },
  { group: "g5", sftData: "SID", margin: "✗", window: "0", special: "—", goal: "单轮 DPO 无 Margin" },
  { group: "g6", sftData: "SID", margin: "✗", window: "2", special: "LoRA 仅 attn 层", goal: "LoRA 作用范围" },
  { group: "g7", sftData: "SID+QA", margin: "✗", window: "2", special: "混入通用 QA", goal: "数据纯度" },
  { group: "g8", sftData: "SID", margin: "✗", window: "2", special: "Qwen-Plus 重写", goal: "候选回复质量" },
  { group: "g9", sftData: "SID", margin: "✓", window: "0", special: "—", goal: "单轮 + Margin" },
  { group: "g10", sftData: "SID", margin: "✓", window: "2", special: "—", goal: "最优配置" },
  { group: "g11", sftData: "SID", margin: "✓", window: "1", special: "—", goal: "窗口对比" },
];

const ABLATION_STAGES = [
  { id: "sft", title: "1. 监督微调预热", subtitle: "SID 数据与 LoRA/SFT 配置", moduleId: "sft_warmup" },
  { id: "simulation", title: "2. 双角色交互仿真", subtitle: "候选教师回复质量", moduleId: "candidate_quality" },
  { id: "evaluation", title: "3. 七维客观评价", subtitle: "统一指标口径", moduleId: "seven_dim" },
  { id: "pairing", title: "4. 对比样本构建", subtitle: "前瞻窗口 w", moduleId: "window" },
  { id: "dpo", title: "5. 偏好对齐优化", subtitle: "连续间隔感知 DPO", moduleId: "margin" },
];

const ABLATION_MODULES: AblationModule[] = [
  {
    id: "sft_warmup",
    stageId: "sft",
    title: "监督微调预热与 SFT 配置",
    description: "比较基线、SFT 预热、无 SFT 直接 DPO、LoRA 作用范围和数据纯度。",
    groups: ["g1", "g2", "g3", "g4", "g6", "g7"],
  },
  {
    id: "candidate_quality",
    stageId: "simulation",
    title: "双角色仿真候选质量",
    description: "以 G4 为参照，比较 Qwen-Plus 重写候选回复后的偏好训练收益。",
    groups: ["g4", "g8"],
  },
  {
    id: "seven_dim",
    stageId: "evaluation",
    title: "G1-G11 七维指标总览",
    description: "观察所有消融组在七维客观评价体系中的整体分布。",
    groups: ABLATION_DESIGNS.map((item) => item.group),
  },
  {
    id: "window",
    stageId: "pairing",
    title: "前瞻窗口影响",
    description: "比较 w=0、w=1、w=2 对结构完整、跨学科迁移和总分的影响。",
    groups: ["g5", "g11", "g10"],
  },
  {
    id: "margin",
    stageId: "dpo",
    title: "连续间隔感知 DPO",
    description: "比较单轮/多轮条件下是否引入连续间隔约束。",
    groups: ["g5", "g9", "g4", "g10"],
  },
  {
    id: "increment",
    stageId: "dpo",
    title: "关键消融路径增量",
    description: "聚焦 G1 → G5 → G10 的关键路径增量。",
    groups: ["g1", "g5", "g10"],
  },
];

const ABLATION_SECTIONS: AblationSection[] = [
  {
    id: "sft",
    title: "监督微调预热实验",
    description: "只比较 SFT 预热、SFT 数据风格或 LoRA 作用范围相关变量。",
    options: [
      { id: "sft_need", label: "是否需要 SFT 预热", description: "G3 与 G4 控制是否先经过 SID 预热", groups: ["g3", "g4"], note: "同为 w=2、无 Margin，变量是是否经过 SID 监督微调预热。" },
      { id: "sft_warmup_context", label: "预热阶段上下文", description: "G1/G2/G3/G4 展示从基线到 SFT+DPO 的路径", groups: ["g1", "g2", "g3", "g4"], note: "这是章节说明用的路径观察，不作为单一变量严格对照。" },
      { id: "lora_scope", label: "LoRA 作用范围", description: "G4 对比 G6", groups: ["g4", "g6"], note: "保持 SID、w=2、无 Margin，变量是 LoRA 是否仅作用 attn 层。" },
      { id: "sft_purity", label: "SFT 数据纯度", description: "G4 对比 G7", groups: ["g4", "g7"], note: "保持 w=2、无 Margin，变量是 SFT 数据从 SID 变为 SID+QA。" },
    ],
  },
  {
    id: "simulation",
    title: "双角色仿真数据质量实验",
    description: "只比较候选教师回复质量是否经过更强模型重写。",
    options: [
      { id: "candidate_quality", label: "候选回复质量", description: "G4 对比 G8", groups: ["g4", "g8"], note: "保持 SID、w=2、无 Margin，变量是是否使用 Qwen-Plus 重写候选回复。" },
    ],
  },
  {
    id: "evaluation",
    title: "七维评价结果分析",
    description: "统一评价口径下观察所有 G 组，不用于解释单一变量因果。",
    options: [
      { id: "seven_dim_overview", label: "G1-G11 总览", description: "所有消融组整体分布", groups: ABLATION_DESIGNS.map((item) => item.group), note: "用于总体观察指标分布和权衡关系，不代表所有组互为严格对照。" },
    ],
  },
  {
    id: "window",
    title: "前瞻窗口影响实验",
    description: "窗口比较必须按 Margin 条件拆开，避免把不同训练目标混成一组。",
    options: [
      { id: "window_no_margin", label: "无 Margin: w=0 vs w=2", description: "G5 对比 G4", groups: ["g5", "g4"], note: "两组都无连续间隔约束，变量是单轮 w=0 与多轮 w=2。" },
      { id: "window_with_margin", label: "有 Margin: w=0 / w=1 / w=2", description: "G9/G11/G10", groups: ["g9", "g11", "g10"], note: "三组都启用 Margin，变量是前瞻窗口大小。" },
    ],
  },
  {
    id: "margin",
    title: "连续间隔感知 DPO 实验",
    description: "Margin 比较必须按单轮与多轮拆开。",
    options: [
      { id: "margin_single", label: "单轮是否启用 Margin", description: "G5 对比 G9", groups: ["g5", "g9"], note: "两组都是 w=0，变量是是否启用连续间隔约束。" },
      { id: "margin_multi", label: "多轮是否启用 Margin", description: "G4 对比 G10", groups: ["g4", "g10"], note: "两组都是 w=2，变量是是否启用连续间隔约束。" },
    ],
  },
  {
    id: "increment",
    title: "关键指标增量分析",
    description: "展示代表性训练路径，不作为单一变量严格对照。",
    options: [
      { id: "best_path", label: "G1 → G5 → G10", description: "基线、单轮 DPO、最优配置", groups: ["g1", "g5", "g10"], note: "这是关键路径增量分析，用于说明主要机制叠加后的整体收益。" },
    ],
  },
];

const ABLATION_OPTION_BY_ID = new Map(
  ABLATION_SECTIONS.flatMap((section) => section.options.map((option) => [option.id, option] as const)),
);

const ABLATION_PRESET_COMPARISON: Record<string, string> = {
  sft_warmup: "sft_need",
  candidate_quality: "candidate_quality",
  seven_dim: "seven_dim_overview",
  window: "window_with_margin",
  margin: "margin_multi",
  increment: "best_path",
};

const totalRows = computed(() => result.value?.rows.length ?? 0);
const importedAtText = computed(() => {
  if (!result.value) return "";
  return new Date(result.value.importedAt).toLocaleString();
});

const sortedSummaries = computed(() => {
  return [...(result.value?.summaries ?? [])].sort((a, b) => {
    const left = a.metrics.TotalScore?.score ?? -Infinity;
    const right = b.metrics.TotalScore?.score ?? -Infinity;
    return right - left;
  });
});

const summaryByModel = computed(() => {
  return new Map(sortedSummaries.value.map((summary) => [summary.modelName.toLowerCase(), summary]));
});

const ablationSummaries = computed(() => {
  return ABLATION_DESIGNS
    .map((design) => summaryByModel.value.get(design.group))
    .filter((summary): summary is MetricSummary => Boolean(summary));
});

const comparisonSummaries = computed(() => {
  return sortedSummaries.value.filter((summary) => !isAblationModelName(summary.modelName));
});

const activeAblationModule = computed(() => {
  return (
    ABLATION_MODULES.find((module) => module.id === selectedAblationModuleId.value) ??
    ABLATION_MODULES[0]
  );
});

const activeAblationStageId = computed(() =>
  selectedAblationModuleId.value === "custom" ? "" : activeAblationModule.value.stageId,
);

const activeAblationTitle = computed(() => {
  return activeAblationComparison.value.label;
});

const activeAblationDescription = computed(() => {
  return activeAblationComparison.value.description;
});

const activeAblationComparison = computed(() => {
  return (
    ABLATION_OPTION_BY_ID.get(selectedAblationComparisonId.value) ??
    ABLATION_SECTIONS[2].options[0]
  );
});

const selectedAblationGroups = computed(() => {
  return [...activeAblationComparison.value.groups];
});

const activeAblationSectionId = computed(() => {
  return (
    ABLATION_SECTIONS.find((section) =>
      section.options.some((option) => option.id === activeAblationComparison.value.id),
    )?.id ?? ABLATION_SECTIONS[2].id
  );
});

const ablationModuleSummaries = computed(() => {
  return selectedAblationGroups.value
    .map((group) => summaryByModel.value.get(group))
    .filter((summary): summary is MetricSummary => Boolean(summary));
});

const selectedChartSourceSummaries = computed(() => {
  if (activeMetricsTab.value === "ablation") return ablationModuleSummaries.value;
  const selected = new Set(selectedChartModels.value);
  const summaries = selected.size
    ? comparisonSummaries.value.filter((summary) => selected.has(summary.modelName))
    : comparisonSummaries.value;
  return summaries;
});

const selectedChartSummaries = computed(() => {
  return selectedChartSourceSummaries.value.slice(0, 8);
});

const chartMetricMax = computed(() => {
  const scores = selectedChartSummaries.value.map(
    (summary) => summary.metrics[chartMetric.value]?.score ?? 0,
  );
  const max = Math.max(...scores, chartMetric.value === "TotalScore" ? 1 : 0);
  return max <= 0 ? 1 : max;
});

const radarMetricKeys = computed(() =>
  METRIC_KEYS.filter((key) => key !== "TotalScore"),
);

const radarAxisMaxByMetric = computed(() => {
  return radarMetricKeys.value.reduce(
    (acc, key) => {
      const maxScore = Math.max(
        1,
        ...selectedChartSummaries.value.map((summary) => summary.metrics[key]?.score ?? 0),
      );
      acc[key] = Math.ceil(maxScore * 4) / 4;
      return acc;
    },
    {} as Record<MetricKey, number>,
  );
});

const chartColors = [
  "#8fd3b0",
  "#f2c27d",
  "#9fc5ff",
  "#f38c8c",
  "#c7a7ff",
  "#b6e6df",
  "#f0a6c8",
  "#d8d8d8",
];

const radarGridRings = [0.25, 0.5, 0.75, 1];
const radarCenter = 180;
const radarRadius = 118;
const radarSize = radarCenter * 2;
const radarAxisPoints = computed(() =>
  radarMetricKeys.value.map((key, index) => ({
    key,
    label: METRIC_LABELS[key],
    max: radarAxisMaxByMetric.value[key] ?? 1,
    labelPoint: radarPoint(index, 1.22),
    maxPoint: radarPoint(index, 1.08),
    ...radarPoint(index, 1),
  })),
);
const radarGridPolygons = computed(() =>
  radarGridRings.map((ratio) => ({
    ratio,
    points: radarMetricKeys.value
      .map((_, index) => {
        const point = radarPoint(index, ratio);
        return `${point.x},${point.y}`;
      })
      .join(" "),
  })),
);
const radarSeries = computed(() =>
  selectedChartSummaries.value.map((summary, index) => ({
    modelName: summary.modelName,
    color: chartColors[index % chartColors.length],
    points: radarMetricKeys.value
      .map((key, metricIndex) => {
        const axisMax = radarAxisMaxByMetric.value[key] ?? 1;
        const score = Math.max(0, summary.metrics[key]?.score ?? 0);
        const point = radarPoint(metricIndex, Math.min(1, score / axisMax));
        return `${point.x},${point.y}`;
      })
      .join(" "),
  })),
);

const filteredRows = computed(() => {
  const keyword = searchText.value.trim().toLowerCase();
  return (result.value?.rows ?? [])
    .filter((row) => {
      if (selectedModel.value && row.modelName !== selectedModel.value) return false;
      if (selectedFile.value && row.fileKey !== selectedFile.value) return false;
      if (!keyword) return true;
      const text = [
        row.modelName,
        row.fileKey,
        row.studentId,
        row.studentType,
        row.scenario,
        row.topicId,
        row.repeatId,
        ...(row.dialogue ?? []).map((turn) => `${turn.role} ${turn.content}`),
        ...row.annotations.map((ann) => `${ann.speaker} ${ann.utterance}`),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return text.includes(keyword);
    })
    .sort((a, b) => {
      const left = a.metrics[selectedMetric.value]?.score ?? -Infinity;
      const right = b.metrics[selectedMetric.value]?.score ?? -Infinity;
      return right - left;
    });
});

const selectedRow = computed(() => {
  return filteredRows.value.find((row) => row.id === selectedRowId.value) ?? filteredRows.value[0] ?? null;
});

const metricRows = computed(() => {
  if (!selectedRow.value) return [];
  return METRIC_KEYS.map((key) => ({
    key,
    label: METRIC_LABELS[key],
    entry: selectedRow.value?.metrics[key],
  }));
});

const detailDialogueTurns = computed(() => {
  const row = selectedRow.value;
  if (!row?.dialogue?.length) return [];
  return row.dialogue.map((turn, index) => ({
    ...turn,
    annotation: row.annotations[index],
    fields: annotationFields(row.annotations[index]),
  }));
});

watch(
  () => selectedRow.value?.id,
  (id) => {
    selectedRowId.value = id ?? "";
  },
  { immediate: true },
);

watch(result, () => {
  selectedModel.value = "";
  selectedFile.value = "";
  selectedMetric.value = "TotalScore";
  chartMetric.value = "TotalScore";
  selectedChartModels.value = comparisonSummaries.value.map((item) => item.modelName);
  searchText.value = "";
  selectedRowId.value = "";
  activeMetricsTab.value = "ablation";
  selectedAblationModuleId.value = "seven_dim";
  selectedAblationComparisonId.value = "seven_dim_overview";
});

function metricDisplay(entry?: MetricEntry) {
  return entry?.display ?? "--";
}

function metricScore(entry?: MetricEntry) {
  return entry?.score ?? 0;
}

function metricBarStyle(entry?: MetricEntry) {
  const width = Math.max(0, Math.min(100, metricScore(entry) * 100));
  return { width: `${width}%` };
}

function chartBarStyle(entry?: MetricEntry) {
  const ratio = metricScore(entry) / chartMetricMax.value;
  return { height: `${Math.max(2, Math.min(100, ratio * 100))}%` };
}

function rowTitle(row: MetricDetailRow) {
  return `${row.fileKey} #${row.rowIndex + 1}`;
}

function rowSubtitle(row: MetricDetailRow) {
  const countText = `对话 ${row.dialogue?.length ?? 0} 条，指标 ${row.metricsComplete ? METRIC_KEYS.length : Object.keys(row.metrics).length} 条`;
  return `${countText} · ${row.metricsSource === "uploaded" ? "上传指标" : row.metricsSource === "computed" ? "标注重算" : "无指标"}`;
}

function rowPreview(row: MetricDetailRow) {
  const firstStudent = row.dialogue?.find((turn) => turn.role === "学生")?.content;
  return firstStudent ?? row.annotations[0]?.utterance ?? "";
}

function groupLabel(group: string) {
  return group.toUpperCase();
}

function getAblationSummary(group: string) {
  return summaryByModel.value.get(group.toLowerCase());
}

function getAblationDesign(group: string) {
  return ABLATION_DESIGNS.find((item) => item.group === group);
}

function selectAblationStage(moduleId: string) {
  selectedAblationModuleId.value = moduleId;
  selectedAblationComparisonId.value =
    ABLATION_PRESET_COMPARISON[moduleId] ?? "seven_dim_overview";
}

function applyAblationPreset(moduleId: string) {
  selectedAblationModuleId.value = moduleId;
  selectedAblationComparisonId.value =
    ABLATION_PRESET_COMPARISON[moduleId] ?? "seven_dim_overview";
}

function selectAblationComparison(option: AblationComparisonOption) {
  selectedAblationComparisonId.value = option.id;
  selectedAblationModuleId.value = "custom";
}

function selectAblationSection(section: AblationSection) {
  if (section.id === activeAblationSectionId.value) return;
  selectAblationComparison(section.options[0]);
}

function dataStateText(group: string) {
  if (getAblationSummary(group)) return "已导入";
  return group === "g3" ? "未收敛" : "无数据";
}

function radarAxisMaxLabel(value: number) {
  return value > 1 ? value.toFixed(2) : "1.00";
}

function annotationFields(annotation?: MetricDetailRow["annotations"][number]) {
  if (!annotation) return [];
  const values = [
    ["意图", annotation.teacher_intent],
    ["策略", annotation.teaching_strategy],
    ["学科", annotation.discipline],
    ["迁移", annotation.discipline_transfer],
    ["状态", annotation.student_cognition_state],
    ["引导", annotation.teacher_guidance_level],
    ["认知", annotation.cognitive_level],
  ] as const;
  return values
    .map(([label, value]) => ({ label, value: value?.trim() }))
    .filter((item) => Boolean(item.value))
    .map((item) => ({ label: item.label, value: item.value as string }));
}

function toggleChartModel(modelName: string) {
  const selected = new Set(selectedChartModels.value);
  if (selected.has(modelName)) {
    selected.delete(modelName);
  } else {
    selected.add(modelName);
  }
  selectedChartModels.value = [...selected];
}

function radarPoint(index: number, ratio: number) {
  const count = Math.max(1, radarMetricKeys.value.length);
  const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
  return {
    x: radarCenter + Math.cos(angle) * radarRadius * ratio,
    y: radarCenter + Math.sin(angle) * radarRadius * ratio,
  };
}

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  await importZip(file);
  input.value = "";
}

async function handleUrlImport() {
  await importZipUrl(zipUrl.value);
}
</script>

<template>
  <section class="metrics-dashboard">
    <div class="metrics-toolbar">
      <div class="upload-group">
        <input ref="fileInput" class="file-input" type="file" accept=".zip,application/zip"
          @change="handleFileChange" />
        <button class="primary-btn" :disabled="loading" @click="fileInput?.click()">
          {{ loading ? '导入中' : '导入 ZIP' }}
        </button>
        <input v-model="zipUrl" class="url-input" type="url" placeholder="https://.../metrics.zip"
          @keyup.enter="handleUrlImport" />
        <button class="secondary-btn" :disabled="loading || !zipUrl.trim()" @click="handleUrlImport">导入 URL</button>
        <button v-if="result" class="secondary-btn" @click="reset">清空</button>
      </div>

      <div class="import-meta">
        <span v-if="result">{{ result.fileName }}</span>
        <span v-if="result">{{ importedAtText }}</span>
      </div>
    </div>

    <div v-if="error" class="metrics-error">{{ error }}</div>

    <div v-if="!result" class="empty-state">
      <div class="empty-title">指标对比</div>
      <div class="empty-subtitle">ZIP: data/final + data/metrics</div>
    </div>

    <template v-else>
      <div class="metrics-tabs" role="tablist" aria-label="指标页面">
        <button class="metrics-tab" :class="{ active: activeMetricsTab === 'ablation' }"
          @click="activeMetricsTab = 'ablation'">
          消融实验
        </button>
        <button class="metrics-tab" :class="{ active: activeMetricsTab === 'comparison' }"
          @click="activeMetricsTab = 'comparison'">
          对比实验
        </button>
        <button class="metrics-tab" :class="{ active: activeMetricsTab === 'details' }"
          @click="activeMetricsTab = 'details'">
          明细
        </button>
      </div>

      <section v-if="activeMetricsTab === 'ablation'" class="tab-panel ablation-panel">
        <div class="summary-strip">
          <div>
            <span class="summary-label">消融组</span>
            <strong>{{ ablationSummaries.length }}/11</strong>
          </div>
          <div>
            <span class="summary-label">当前模块</span>
            <strong>{{ selectedAblationGroups.length }}</strong>
          </div>
          <div>
            <span class="summary-label">模块数据</span>
            <strong>{{ ablationModuleSummaries.length }}</strong>
          </div>
        </div>

        <section class="ablation-stage-board" aria-label="按消融章节选择对比问题">
          <article v-for="(section, sectionIndex) in ABLATION_SECTIONS" :key="section.id" class="ablation-stage-card"
            :class="{ active: activeAblationSectionId === section.id }" @click="selectAblationSection(section)">
            <div class="section-copy">
              <span class="section-index">{{ sectionIndex + 1 }}</span>
              <h3>{{ section.title }}</h3>
              <p>{{ section.description }}</p>
            </div>
            <div class="comparison-options">
              <button v-for="option in section.options" :key="option.id" class="comparison-option"
                :class="{ active: selectedAblationComparisonId === option.id }"
                @click.stop="selectAblationComparison(option)">
                <span class="switch-copy">
                  <strong>{{ option.label }}</strong>
                  <small>{{ option.description }}</small>
                </span>
                <span class="switch-groups">{{ option.groups.map(groupLabel).join(" ") }}</span>
              </button>
            </div>
          </article>
        </section>

        <section class="ablation-design-grid" aria-label="当前消融组说明">
          <article v-for="group in selectedAblationGroups" :key="group" class="ablation-design-card"
            :class="{ imported: Boolean(getAblationSummary(group)) }">
            <div class="design-card-head">
              <strong>{{ groupLabel(group) }}</strong>
              <span>{{ dataStateText(group) }}</span>
            </div>
            <p>{{ getAblationDesign(group)?.goal ?? "未定义" }}</p>
            <div class="design-tags">
              <span>SFT {{ getAblationDesign(group)?.sftData ?? '—' }}</span>
              <span>Margin {{ getAblationDesign(group)?.margin ?? '—' }}</span>
              <span>w={{ getAblationDesign(group)?.window ?? '—' }}</span>
              <span>{{ getAblationDesign(group)?.special ?? '标准配置' }}</span>
            </div>
          </article>
        </section>

        <section class="chart-panel">
          <div class="chart-controls">
            <div class="module-copy">
              <h3>{{ activeAblationTitle }}</h3>
              <p>{{ activeAblationDescription }}</p>
              <p class="comparison-note">{{ activeAblationComparison.note }}</p>
            </div>
            <select v-model="chartMetric">
              <option v-for="key in METRIC_KEYS" :key="key" :value="key">{{ METRIC_LABELS[key] }}</option>
            </select>
          </div>

          <div class="charts-grid">
            <div class="chart-card">
              <div class="chart-title">
                <span>{{ METRIC_LABELS[chartMetric] }}</span>
                <small>{{ selectedChartSummaries.length }} 个实验组</small>
              </div>
              <div class="bar-chart">
                <div v-for="(summary, index) in selectedChartSummaries" :key="summary.modelName" class="bar-item">
                  <div class="bar-track">
                    <span class="bar-fill" :style="{ ...chartBarStyle(summary.metrics[chartMetric]), background: chartColors[index % chartColors.length] }"></span>
                  </div>
                  <span class="bar-value">{{ metricDisplay(summary.metrics[chartMetric]) }}</span>
                  <span class="bar-label" :title="summary.modelName">{{ groupLabel(summary.modelName) }}</span>
                </div>
              </div>
            </div>

            <div class="chart-card radar-card">
              <div class="chart-title">
                <span>七维雷达</span>
                <small>不含总分</small>
              </div>
              <div class="radar-wrap">
                <svg class="radar-chart" :viewBox="`0 0 ${radarSize} ${radarSize}`" role="img">
                  <polygon v-for="polygon in radarGridPolygons" :key="polygon.points" class="radar-grid"
                    :class="{ outer: polygon.ratio === 1 }" :points="polygon.points" />
                  <line v-for="axis in radarAxisPoints" :key="axis.key" class="radar-axis" :x1="radarCenter"
                    :y1="radarCenter" :x2="axis.x" :y2="axis.y" />
                  <text v-for="axis in radarAxisPoints" :key="`${axis.key}-label`" class="radar-label"
                    :x="axis.labelPoint.x" :y="axis.labelPoint.y" text-anchor="middle" dominant-baseline="middle">
                    {{ axis.label }}
                  </text>
                  <text v-for="axis in radarAxisPoints" :key="`${axis.key}-max`" class="radar-score-label"
                    :x="axis.maxPoint.x" :y="axis.maxPoint.y" text-anchor="middle" dominant-baseline="middle">
                    {{ radarAxisMaxLabel(axis.max) }}
                  </text>
                  <polygon v-for="series in radarSeries" :key="series.modelName" class="radar-series"
                    :points="series.points" :stroke="series.color" />
                </svg>
                <div class="radar-legend">
                  <span v-for="(summary, index) in selectedChartSummaries" :key="summary.modelName">
                    <i :style="{ background: chartColors[index % chartColors.length] }"></i>
                    {{ groupLabel(summary.modelName) }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div class="comparison-table-wrap">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>组号</th>
                <th>SFT 数据</th>
                <th>Margin</th>
                <th>窗口 w</th>
                <th>特殊配置</th>
                <th>验证目标</th>
                <th v-for="key in METRIC_KEYS" :key="key">{{ METRIC_LABELS[key] }}</th>
                <th>数据</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="group in selectedAblationGroups" :key="group">
                <td class="model-cell">{{ groupLabel(group) }}</td>
                <td>{{ getAblationDesign(group)?.sftData ?? '—' }}</td>
                <td>{{ getAblationDesign(group)?.margin ?? '—' }}</td>
                <td>{{ getAblationDesign(group)?.window ?? '—' }}</td>
                <td>{{ getAblationDesign(group)?.special ?? '—' }}</td>
                <td>{{ getAblationDesign(group)?.goal ?? '—' }}</td>
                <td v-for="key in METRIC_KEYS" :key="key">
                  <span class="metric-display">{{ metricDisplay(getAblationSummary(group)?.metrics[key]) }}</span>
                </td>
                <td>{{ dataStateText(group) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-else-if="activeMetricsTab === 'comparison'" class="tab-panel overview-panel">
        <div class="summary-strip">
          <div>
            <span class="summary-label">模型</span>
            <strong>{{ comparisonSummaries.length }}</strong>
          </div>
          <div>
            <span class="summary-label">记录</span>
            <strong>{{ totalRows }}</strong>
          </div>
          <div>
            <span class="summary-label">文件</span>
            <strong>{{ fileOptions.length }}</strong>
          </div>
        </div>

        <section class="chart-panel">
          <div class="chart-controls">
            <div class="model-chips" aria-label="选择图表模型">
              <button v-for="summary in comparisonSummaries" :key="summary.modelName" class="model-chip"
                :class="{ selected: selectedChartModels.includes(summary.modelName) }"
                @click="toggleChartModel(summary.modelName)">
                {{ summary.modelName }}
              </button>
            </div>
            <select v-model="chartMetric">
              <option v-for="key in METRIC_KEYS" :key="key" :value="key">{{ METRIC_LABELS[key] }}</option>
            </select>
          </div>

          <div class="charts-grid">
            <div class="chart-card">
              <div class="chart-title">
                <span>{{ METRIC_LABELS[chartMetric] }}</span>
              <small>{{ selectedChartSummaries.length }} 个模型</small>
              </div>
              <div class="bar-chart">
                <div v-for="(summary, index) in selectedChartSummaries" :key="summary.modelName" class="bar-item">
                  <div class="bar-track">
                    <span class="bar-fill" :style="{ ...chartBarStyle(summary.metrics[chartMetric]), background: chartColors[index % chartColors.length] }"></span>
                  </div>
                  <span class="bar-value">{{ metricDisplay(summary.metrics[chartMetric]) }}</span>
                  <span class="bar-label" :title="summary.modelName">{{ summary.modelName }}</span>
                </div>
              </div>
            </div>

            <div class="chart-card radar-card">
              <div class="chart-title">
                <span>全指标雷达</span>
                <small>不含总分</small>
              </div>
              <div class="radar-wrap">
                <svg class="radar-chart" :viewBox="`0 0 ${radarSize} ${radarSize}`" role="img">
                  <polygon v-for="polygon in radarGridPolygons" :key="polygon.points" class="radar-grid"
                    :class="{ outer: polygon.ratio === 1 }" :points="polygon.points" />
                  <line v-for="axis in radarAxisPoints" :key="axis.key" class="radar-axis" :x1="radarCenter"
                    :y1="radarCenter" :x2="axis.x" :y2="axis.y" />
                  <text v-for="axis in radarAxisPoints" :key="`${axis.key}-label`" class="radar-label"
                    :x="axis.labelPoint.x" :y="axis.labelPoint.y" text-anchor="middle" dominant-baseline="middle">
                    {{ axis.label }}
                  </text>
                  <text v-for="axis in radarAxisPoints" :key="`${axis.key}-max`" class="radar-score-label"
                    :x="axis.maxPoint.x" :y="axis.maxPoint.y" text-anchor="middle" dominant-baseline="middle">
                    {{ radarAxisMaxLabel(axis.max) }}
                  </text>
                  <polygon v-for="series in radarSeries" :key="series.modelName" class="radar-series"
                    :points="series.points" :stroke="series.color" />
                </svg>
                <div class="radar-legend">
                  <span v-for="(summary, index) in selectedChartSummaries" :key="summary.modelName">
                    <i :style="{ background: chartColors[index % chartColors.length] }"></i>
                    {{ summary.modelName }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div class="comparison-table-wrap">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>模型</th>
                <th>来源</th>
                <th>数量</th>
                <th v-for="key in METRIC_KEYS" :key="key">{{ METRIC_LABELS[key] }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="summary in comparisonSummaries" :key="summary.modelName"
                :class="{ selected: selectedModel === summary.modelName }"
                @click="selectedModel = selectedModel === summary.modelName ? '' : summary.modelName">
                <td class="model-cell">{{ summary.modelName }}</td>
                <td>{{ summary.source === 'uploaded' ? '上传' : '重算' }}</td>
                <td>{{ summary.totalDialogues }}</td>
                <td v-for="key in METRIC_KEYS" :key="key">
                  <span class="metric-display">{{ metricDisplay(summary.metrics[key]) }}</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-else-if="activeMetricsTab === 'details'" class="tab-panel details-panel">
        <div class="metrics-body">
          <aside class="detail-list">
            <div class="filters">
              <select v-model="selectedModel">
                <option value="">全部模型</option>
                <option v-for="model in modelOptions" :key="model" :value="model">{{ model }}</option>
              </select>
              <select v-model="selectedFile">
                <option value="">全部文件</option>
                <option v-for="file in fileOptions" :key="file" :value="file">{{ file }}</option>
              </select>
              <select v-model="selectedMetric">
                <option v-for="key in METRIC_KEYS" :key="key" :value="key">{{ METRIC_LABELS[key] }}</option>
              </select>
              <input v-model="searchText" type="search" placeholder="搜索" />
            </div>

            <div class="row-list">
              <button v-for="row in filteredRows" :key="row.id" class="row-item"
                :class="{ selected: row.id === selectedRow?.id }" @click="selectedRowId = row.id">
                <span class="row-main">
                  <span class="row-model">{{ row.modelName }}</span>
                  <span class="row-file">{{ rowTitle(row) }}</span>
                </span>
                <span class="row-score">{{ metricDisplay(row.metrics[selectedMetric]) }}</span>
                <span class="row-source">{{ rowSubtitle(row) }}</span>
                <span v-if="rowPreview(row)" class="row-preview">{{ rowPreview(row) }}</span>
              </button>
            </div>
          </aside>

          <article v-if="selectedRow" class="detail-panel">
            <header class="detail-header">
              <div>
                <h2>{{ selectedRow.modelName }}</h2>
                <p>{{ rowTitle(selectedRow) }} · {{ selectedRow.studentId || '无 student_id' }}</p>
              </div>
              <div class="total-score">{{ metricDisplay(selectedRow.metrics.TotalScore) }}</div>
            </header>

            <div class="metric-grid">
              <div v-for="item in metricRows" :key="item.key" class="metric-tile">
                <div class="metric-tile-head">
                  <span>{{ item.label }}</span>
                  <strong>{{ metricDisplay(item.entry) }}</strong>
                </div>
                <div class="metric-bar">
                  <span :style="metricBarStyle(item.entry)"></span>
                </div>
                <small v-if="item.entry?.rawValue !== undefined">raw {{ item.entry.rawValue }}</small>
              </div>
            </div>

            <div class="detail-stack">
              <section class="dialogue-section">
                <h3>对话内容</h3>
                <div v-if="detailDialogueTurns.length" class="dialogue-list">
                  <div v-for="(turn, index) in detailDialogueTurns" :key="`${turn.role}-${index}`"
                    class="dialogue-row" :class="turn.role === '教师' ? 'role-teacher' : 'role-student'">
                    <div class="turn-avatar">{{ turn.role === '教师' ? '师' : '生' }}</div>
                    <div class="turn-content">
                      <div class="dialogue-turn">
                        <span class="turn-role">{{ turn.role }}</span>
                        <p>{{ turn.content }}</p>
                      </div>
                      <div v-if="turn.fields.length" class="turn-annotations">
                        <span v-for="field in turn.fields" :key="`${index}-${field.label}`" class="turn-chip">
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                <div v-else class="empty-inline">无对话内容</div>
              </section>

              <section class="annotation-section">
                <h3>标注</h3>
                <div v-if="selectedRow.annotations.length" class="annotation-table-wrap">
                  <table class="annotation-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>角色</th>
                        <th>意图</th>
                        <th>策略</th>
                        <th>学科</th>
                        <th>迁移</th>
                        <th>状态</th>
                        <th>引导</th>
                        <th>认知</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(ann, index) in selectedRow.annotations" :key="`${ann.speaker}-${index}`">
                        <td>{{ index + 1 }}</td>
                        <td>{{ ann.speaker }}</td>
                        <td>{{ ann.teacher_intent || '--' }}</td>
                        <td>{{ ann.teaching_strategy || '--' }}</td>
                        <td>{{ ann.discipline || '--' }}</td>
                        <td>{{ ann.discipline_transfer || '--' }}</td>
                        <td>{{ ann.student_cognition_state || '--' }}</td>
                        <td>{{ ann.teacher_guidance_level || '--' }}</td>
                        <td>{{ ann.cognitive_level || '--' }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div v-else class="empty-inline">无标注</div>
              </section>
            </div>
          </article>
        </div>
      </section>

    </template>
  </section>
</template>

<style scoped>
.metrics-dashboard {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
  overflow: hidden;
  padding: 18px;
}

.metrics-toolbar {
  align-items: center;
  display: flex;
  flex-shrink: 0;
  gap: 14px;
  justify-content: space-between;
}

.upload-group,
.import-meta {
  align-items: center;
  display: flex;
  gap: 8px;
  min-width: 0;
}

.upload-group {
  flex: 1 1 auto;
  flex-wrap: wrap;
}

.file-input {
  display: none;
}

.primary-btn,
.secondary-btn {
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.84rem;
  min-height: 34px;
  padding: 7px 12px;
}

.url-input {
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  flex: 1 1 280px;
  font-family: var(--font-sans);
  font-size: 0.82rem;
  min-height: 34px;
  min-width: 220px;
  padding: 7px 10px;
}

.url-input::placeholder {
  color: var(--text-tertiary);
}

.primary-btn {
  background: #f2f2f2;
  border: 1px solid transparent;
  color: #191919;
  font-weight: 650;
}

.primary-btn:disabled {
  cursor: not-allowed;
  opacity: 0.65;
}

.secondary-btn {
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  color: var(--text-primary);
}

.secondary-btn:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.import-meta {
  color: var(--text-tertiary);
  font-size: 0.78rem;
  justify-content: flex-end;
  overflow: hidden;
}

.import-meta span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.metrics-error,
.empty-state,
.empty-inline {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
}

.metrics-error {
  background: rgba(243, 140, 140, 0.12);
  color: #ffc6c6;
  flex-shrink: 0;
  padding: 10px 12px;
}

.empty-state {
  align-items: center;
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  min-height: 260px;
}

.empty-title {
  color: var(--text-primary);
  font-size: 1.35rem;
  font-weight: 650;
}

.empty-subtitle {
  color: var(--text-tertiary);
  font-size: 0.86rem;
  margin-top: 6px;
}

.metrics-tabs {
  align-items: center;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  display: flex;
  flex-shrink: 0;
  gap: 3px;
  padding: 3px;
  width: fit-content;
}

.metrics-tab {
  align-items: center;
  background: transparent;
  border: none;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  display: inline-flex;
  font-family: var(--font-sans);
  font-size: 0.84rem;
  gap: 6px;
  min-height: 32px;
  padding: 6px 13px;
  white-space: nowrap;
}

.metrics-tab:hover {
  color: var(--text-primary);
}

.metrics-tab.active {
  background: #f2f2f2;
  color: #191919;
  font-weight: 650;
}

.metrics-tab span {
  border: 1px solid rgba(242, 194, 125, 0.42);
  border-radius: var(--radius-full);
  color: inherit;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  line-height: 1;
  padding: 2px 5px;
}

.tab-panel {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
}

.overview-panel,
.details-panel,
.ablation-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.ablation-stage-board {
  display: grid;
  flex-shrink: 0;
  gap: 10px;
  grid-template-columns: repeat(6, minmax(150px, 1fr));
}

.module-copy {
  min-width: 0;
}

.module-copy h3 {
  color: var(--text-primary);
  font-size: 0.96rem;
  font-weight: 700;
  line-height: 1.25;
}

.module-copy p {
  color: var(--text-tertiary);
  font-size: 0.78rem;
  line-height: 1.45;
  margin-top: 4px;
}

.module-copy .comparison-note {
  color: var(--accent-warning);
  font-size: 0.74rem;
  margin-top: 5px;
}

.ablation-stage-card {
  background: rgba(255, 255, 255, 0.032);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  display: grid;
  gap: 8px;
  grid-template-rows: 88px 1fr;
  min-width: 0;
  min-height: 260px;
  padding: 12px;
  position: relative;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.ablation-stage-card:not(:last-child)::after {
  color: var(--text-tertiary);
  content: "→";
  font-family: var(--font-mono);
  font-size: 1rem;
  position: absolute;
  right: -9px;
  top: 28px;
  z-index: 2;
}

.ablation-stage-card.active {
  background: rgba(143, 211, 176, 0.1);
  border-color: rgba(143, 211, 176, 0.42);
  box-shadow: inset 0 0 0 1px rgba(143, 211, 176, 0.18), 0 10px 24px rgba(0, 0, 0, 0.22);
  z-index: 3;
}

.section-copy {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.section-index {
  align-items: center;
  background: rgba(242, 194, 125, 0.14);
  border: 1px solid rgba(242, 194, 125, 0.35);
  border-radius: var(--radius-full);
  color: var(--accent-warning);
  display: inline-flex;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  height: 22px;
  justify-content: center;
  width: 22px;
}

.section-copy h3 {
  color: var(--text-primary);
  font-size: 0.86rem;
  font-weight: 700;
  line-height: 1.3;
  min-height: 34px;
}

.section-copy p {
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  color: var(--text-tertiary);
  display: -webkit-box;
  font-size: 0.72rem;
  line-height: 1.35;
  min-height: 38px;
  overflow: hidden;
}

.comparison-options {
  display: grid;
  gap: 6px;
  grid-auto-rows: 76px;
}

.comparison-option {
  align-items: center;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  display: grid;
  gap: 5px;
  min-height: 76px;
  padding: 9px 10px;
  text-align: left;
}

.comparison-option:hover,
.comparison-option.active {
  background: rgba(143, 211, 176, 0.1);
  border-color: rgba(143, 211, 176, 0.42);
  color: var(--text-primary);
}

.switch-copy {
  display: grid;
  gap: 2px;
  grid-template-rows: auto 1fr;
  min-width: 0;
}

.switch-copy strong {
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 1;
  color: var(--text-primary);
  display: -webkit-box;
  font-size: 0.82rem;
  line-height: 1.25;
  overflow: hidden;
}

.switch-copy small {
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  color: var(--text-tertiary);
  display: -webkit-box;
  font-size: 0.72rem;
  line-height: 1.25;
  min-height: 36px;
  overflow: hidden;
}

.switch-groups {
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  line-height: 1.25;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ablation-design-grid {
  display: grid;
  flex-shrink: 0;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.ablation-design-card {
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  display: grid;
  gap: 9px;
  min-width: 0;
  padding: 11px;
}

.ablation-design-card.imported {
  border-color: rgba(143, 211, 176, 0.28);
}

.design-card-head {
  align-items: center;
  display: flex;
  gap: 8px;
  justify-content: space-between;
}

.design-card-head strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.95rem;
}

.design-card-head span {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-tertiary);
  font-size: 0.68rem;
  padding: 2px 7px;
}

.ablation-design-card.imported .design-card-head span {
  border-color: rgba(143, 211, 176, 0.38);
  color: var(--accent-success);
}

.ablation-design-card p {
  color: var(--text-secondary);
  font-size: 0.82rem;
  font-weight: 650;
  line-height: 1.35;
}

.design-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.design-tags span {
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-tertiary);
  font-size: 0.68rem;
  line-height: 1.2;
  padding: 3px 7px;
}

.summary-strip {
  display: grid;
  flex-shrink: 0;
  gap: 10px;
  grid-template-columns: repeat(3, minmax(120px, 1fr));
}

.summary-strip>div {
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
}

.summary-label {
  color: var(--text-tertiary);
  display: block;
  font-size: 0.72rem;
}

.summary-strip strong {
  color: var(--text-primary);
  display: block;
  font-size: 1.2rem;
  line-height: 1.2;
  margin-top: 2px;
}

.chart-panel {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  flex-shrink: 0;
  padding: 12px;
}

.chart-controls {
  align-items: flex-start;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  margin-bottom: 12px;
}

.model-chips {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  gap: 7px;
  min-width: 0;
}

.model-chip {
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.75rem;
  line-height: 1.2;
  max-width: 190px;
  min-height: 28px;
  overflow: hidden;
  padding: 5px 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-chip.selected {
  background: rgba(143, 211, 176, 0.14);
  border-color: rgba(143, 211, 176, 0.42);
  color: var(--text-primary);
}

.chart-controls select {
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  flex: 0 0 150px;
  font-family: var(--font-sans);
  font-size: 0.82rem;
  min-height: 32px;
  padding: 6px 8px;
}

.charts-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
}

.chart-card {
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  min-width: 0;
  padding: 12px;
}

.chart-title {
  align-items: baseline;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.chart-title span {
  color: var(--text-primary);
  font-size: 0.9rem;
  font-weight: 650;
}

.chart-title small {
  color: var(--text-tertiary);
  font-size: 0.72rem;
}

.bar-chart {
  align-items: end;
  display: grid;
  gap: 10px;
  grid-auto-flow: column;
  grid-auto-columns: minmax(72px, 1fr);
  min-height: 172px;
  overflow-x: auto;
  padding-bottom: 2px;
}

.bar-item {
  align-items: center;
  display: grid;
  gap: 6px;
  grid-template-rows: 120px auto auto;
  min-width: 72px;
}

.bar-track {
  align-items: end;
  align-self: stretch;
  background: rgba(255, 255, 255, 0.07);
  border-radius: var(--radius-sm);
  display: flex;
  justify-content: center;
  overflow: hidden;
  width: 100%;
}

.bar-fill {
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  display: block;
  min-height: 2px;
  width: 100%;
}

.bar-value {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.76rem;
  font-weight: 700;
  line-height: 1.15;
  text-align: center;
}

.bar-label {
  color: var(--text-tertiary);
  font-size: 0.7rem;
  line-height: 1.2;
  max-width: 96px;
  overflow: hidden;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.radar-card {
  min-height: 224px;
}

.radar-wrap {
  align-items: stretch;
  display: grid;
  gap: 10px;
  grid-template-columns: 340px minmax(0, 1fr);
}

.radar-chart {
  height: 320px;
  width: 320px;
}

.radar-grid {
  fill: none;
  stroke: rgba(255, 255, 255, 0.18);
  stroke-width: 1;
}

.radar-grid.outer {
  stroke: rgba(255, 255, 255, 0.42);
  stroke-width: 1.8;
}

.radar-axis {
  stroke: rgba(255, 255, 255, 0.2);
  stroke-width: 1;
}

.radar-label {
  fill: var(--text-secondary);
  font-family: var(--font-sans);
  font-size: 11px;
  font-weight: 650;
}

.radar-score-label {
  fill: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 10px;
}

.radar-series {
  fill: none;
  opacity: 0.9;
  stroke-linejoin: round;
  stroke-width: 2.4;
}

.radar-legend {
  display: flex;
  flex-direction: column;
  gap: 6px;
  justify-content: center;
  min-width: 0;
}

.radar-legend span {
  align-items: center;
  color: var(--text-secondary);
  display: flex;
  font-size: 0.72rem;
  gap: 6px;
  line-height: 1.25;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.radar-legend i {
  border-radius: var(--radius-full);
  display: block;
  flex: 0 0 auto;
  height: 8px;
  width: 8px;
}

.comparison-table-wrap,
.annotation-table-wrap {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  overflow: auto;
}

.comparison-table-wrap {
  flex-shrink: 0;
  max-height: 236px;
}

.comparison-table,
.annotation-table {
  border-collapse: collapse;
  min-width: 960px;
  width: 100%;
}

.comparison-table th,
.comparison-table td,
.annotation-table th,
.annotation-table td {
  border-bottom: 1px solid var(--border-primary);
  color: var(--text-secondary);
  font-size: 0.78rem;
  padding: 8px 10px;
  text-align: left;
  white-space: nowrap;
}

.comparison-table th,
.annotation-table th {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-tertiary);
  font-weight: 600;
  position: sticky;
  top: 0;
  z-index: 1;
}

.comparison-table tbody tr {
  cursor: pointer;
}

.comparison-table tbody tr:hover,
.comparison-table tbody tr.selected {
  background: rgba(255, 255, 255, 0.055);
}

.model-cell,
.metric-display {
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.metrics-body {
  display: grid;
  flex: 1;
  gap: 14px;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  min-height: 0;
}

.detail-list,
.detail-panel {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  min-height: 0;
  overflow: hidden;
}

.detail-list {
  display: flex;
  flex-direction: column;
}

.filters {
  border-bottom: 1px solid var(--border-primary);
  display: grid;
  flex-shrink: 0;
  gap: 8px;
  grid-template-columns: 1fr;
  padding: 10px;
}

.filters select,
.filters input {
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 0.82rem;
  min-height: 34px;
  padding: 7px 9px;
}

.row-list {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
  overflow-y: auto;
  padding: 8px;
}

.row-item {
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  display: grid;
  grid-template-areas:
    "main score"
    "source score"
    "preview preview";
  grid-template-columns: minmax(0, 1fr) auto;
  font-family: var(--font-sans);
  gap: 4px 10px;
  line-height: 1.35;
  padding: 10px;
  text-align: left;
}

.row-item:hover,
.row-item.selected {
  background: rgba(255, 255, 255, 0.055);
  border-color: var(--border-primary);
}

.row-main {
  align-items: center;
  display: flex;
  gap: 8px;
  grid-area: main;
  min-width: 0;
}

.row-model {
  color: var(--text-primary);
  font-weight: 650;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-file,
.row-source {
  color: var(--text-tertiary);
  font-size: 0.72rem;
  line-height: 1.3;
}

.row-file {
  flex: 0 1 auto;
  min-width: 72px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-source {
  grid-area: source;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-score {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.95rem;
  font-weight: 700;
  grid-area: score;
  justify-self: end;
  line-height: 1.2;
  white-space: nowrap;
}

.row-preview {
  color: var(--text-secondary);
  display: -webkit-box;
  font-size: 0.76rem;
  grid-area: preview;
  line-height: 1.45;
  max-height: 2.9em;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.detail-panel {
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 16px;
}

.detail-header {
  align-items: flex-start;
  display: flex;
  flex-shrink: 0;
  gap: 16px;
  justify-content: space-between;
  margin-bottom: 14px;
}

.detail-header h2 {
  color: var(--text-primary);
  font-size: 1.08rem;
  font-weight: 650;
  line-height: 1.25;
}

.detail-header p {
  color: var(--text-tertiary);
  font-size: 0.78rem;
  margin-top: 4px;
}

.total-score {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 1.45rem;
  font-weight: 750;
}

.metric-grid {
  display: grid;
  flex-shrink: 0;
  gap: 10px;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  margin-bottom: 16px;
}

.metric-tile {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  min-width: 0;
  padding: 10px;
}

.metric-tile-head {
  align-items: baseline;
  display: flex;
  gap: 8px;
  justify-content: space-between;
}

.metric-tile span {
  color: var(--text-tertiary);
  font-size: 0.72rem;
}

.metric-tile strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.82rem;
}

.metric-bar {
  background: rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-full);
  height: 5px;
  margin-top: 9px;
  overflow: hidden;
}

.metric-bar span {
  background: var(--accent-success);
  display: block;
  height: 100%;
}

.metric-tile small {
  color: var(--text-tertiary);
  display: block;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  margin-top: 6px;
}

.detail-stack {
  display: grid;
  flex: 0 0 auto;
  gap: 16px;
  grid-template-columns: minmax(0, 1fr);
}

.dialogue-section h3,
.annotation-section h3 {
  color: var(--text-primary);
  font-size: 0.95rem;
  font-weight: 650;
  margin-bottom: 10px;
}

.dialogue-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.dialogue-row {
  align-items: flex-start;
  display: flex;
  gap: 10px;
  max-width: 100%;
}

.dialogue-row.role-student {
  flex-direction: row-reverse;
}

.turn-avatar {
  align-items: center;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  display: flex;
  flex: 0 0 auto;
  font-size: 0.72rem;
  font-weight: 700;
  height: 30px;
  justify-content: center;
  margin-top: 4px;
  width: 30px;
}

.role-teacher .turn-avatar {
  border-color: rgba(143, 211, 176, 0.3);
  color: var(--accent-success);
}

.turn-content {
  display: flex;
  flex: 0 1 auto;
  flex-direction: column;
  gap: 7px;
  max-width: min(860px, 82%);
  min-width: 0;
}

.role-student .turn-content {
  align-items: flex-end;
}

.dialogue-turn {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 10px;
  width: 100%;
}

.role-teacher .dialogue-turn {
  border-color: rgba(143, 211, 176, 0.26);
}

.role-student .dialogue-turn {
  background: rgba(255, 255, 255, 0.075);
}

.turn-role {
  color: var(--text-tertiary);
  display: block;
  font-size: 0.72rem;
  margin-bottom: 5px;
}

.dialogue-turn p {
  color: var(--text-primary);
  font-size: 0.86rem;
  line-height: 1.72;
  white-space: pre-wrap;
  word-break: break-word;
}

.turn-annotations {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-width: 100%;
}

.role-student .turn-annotations {
  justify-content: flex-end;
}

.turn-chip {
  align-items: center;
  background: rgba(255, 255, 255, 0.055);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  display: inline-flex;
  font-size: 0.72rem;
  gap: 5px;
  line-height: 1.2;
  max-width: 280px;
  min-height: 25px;
  min-width: 0;
  padding: 4px 8px;
}

.turn-chip span {
  color: var(--text-tertiary);
  flex: 0 0 auto;
}

.turn-chip strong {
  color: var(--text-primary);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.annotation-table {
  min-width: 860px;
}

.empty-inline {
  padding: 14px;
}

@media (max-width: 1180px) {
  .metrics-body,
  .detail-stack,
  .charts-grid {
    grid-template-columns: 1fr;
  }

  .ablation-stage-board {
    grid-template-columns: repeat(2, minmax(220px, 1fr));
  }

  .ablation-stage-card:not(:last-child)::after {
    display: none;
  }

  .detail-list {
    max-height: 360px;
  }

  .metric-grid {
    grid-template-columns: repeat(2, minmax(130px, 1fr));
  }

  .radar-wrap {
    grid-template-columns: 340px minmax(0, 1fr);
  }
}

@media (max-width: 720px) {
  .metrics-dashboard {
    padding: 12px;
  }

  .metrics-toolbar,
  .import-meta,
  .chart-controls {
    align-items: stretch;
    flex-direction: column;
  }

  .summary-strip,
  .metric-grid,
  .ablation-stage-board {
    grid-template-columns: 1fr;
  }

  .chart-controls select {
    flex-basis: auto;
  }

  .radar-wrap {
    grid-template-columns: 1fr;
    justify-items: center;
  }

  .turn-content {
    max-width: calc(100% - 40px);
  }
}
</style>
