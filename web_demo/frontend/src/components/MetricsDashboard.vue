<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  METRIC_KEYS,
  METRIC_LABELS,
  useMetricsExplorer,
  type MetricDetailRow,
  type MetricEntry,
  type MetricKey,
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
const activeMetricsTab = ref<"overview" | "details">("overview");

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

const selectedChartSummaries = computed(() => {
  const selected = new Set(selectedChartModels.value);
  const summaries = selected.size
    ? sortedSummaries.value.filter((summary) => selected.has(summary.modelName))
    : sortedSummaries.value.slice(0, 5);
  return summaries.slice(0, 8);
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
    labelPoint: radarPoint(index, 1.22),
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
        const score = Math.max(0, Math.min(1, summary.metrics[key]?.score ?? 0));
        const point = radarPoint(metricIndex, score);
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
  selectedChartModels.value = sortedSummaries.value.slice(0, 4).map((item) => item.modelName);
  searchText.value = "";
  selectedRowId.value = "";
  activeMetricsTab.value = "overview";
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
        <button class="metrics-tab" :class="{ active: activeMetricsTab === 'overview' }"
          @click="activeMetricsTab = 'overview'">
          概览
        </button>
        <button class="metrics-tab" :class="{ active: activeMetricsTab === 'details' }"
          @click="activeMetricsTab = 'details'">
          明细
        </button>
      </div>

      <section v-if="activeMetricsTab === 'overview'" class="tab-panel overview-panel">
        <div class="summary-strip">
          <div>
            <span class="summary-label">模型</span>
            <strong>{{ sortedSummaries.length }}</strong>
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
              <button v-for="summary in sortedSummaries" :key="summary.modelName" class="model-chip"
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
                  <text class="radar-score-label" :x="radarCenter + 4" :y="radarCenter - radarRadius + 12">100%</text>
                  <polyline v-for="series in radarSeries" :key="series.modelName" class="radar-series"
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
              <tr v-for="summary in sortedSummaries" :key="summary.modelName"
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
.details-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
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
  .metric-grid {
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
