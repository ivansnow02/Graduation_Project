<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import type { EvaluationRecord } from '../composables/useEvaluation'

const props = defineProps<{
  collapsed: boolean
  result: EvaluationRecord | null
  running: boolean
  error: string
  stale: boolean
  canRun: boolean
}>()

defineEmits<{
  run: []
  toggleCollapse: []
}>()

const showAnnotations = shallowRef(false)

const metricRows = computed(() => {
  if (!props.result) return []
  return [
    ['StrategyDensity', '策略密度', props.result.metrics.StrategyDensity],
    ['StrategyVariety', '策略多样性', props.result.metrics.StrategyVariety],
    ['IKT', '跨学科迁移', props.result.metrics.IKT],
    ['BP', 'Bloom 进阶', props.result.metrics.BP],
    ['StructureCompleteness', '结构完整度', props.result.metrics.StructureCompleteness],
    ['L3GuidanceRate', 'L3 引导率', props.result.metrics.L3GuidanceRate],
    ['CognitiveCorrectionRate', '认知纠偏率', props.result.metrics.CognitiveCorrectionRate],
  ] as const
})

const evaluatedAtText = computed(() => {
  if (!props.result) return ''
  return new Date(props.result.evaluatedAt).toLocaleString()
})

const totalScoreText = computed(() => props.result?.metrics.TotalScore.display ?? '--')
</script>

<template>
  <aside class="evaluation-panel" :class="{ collapsed }">
    <div class="evaluation-header">
      <div>
        <h2 class="evaluation-title">客观评测</h2>
        <p class="evaluation-subtitle">当前单模型评测会话</p>
      </div>
      <div class="evaluation-actions">
        <button
          class="collapse-btn"
          :aria-label="collapsed ? '展开评测面板' : '折叠评测面板'"
          :title="collapsed ? '展开评测面板' : '折叠评测面板'"
          @click="$emit('toggleCollapse')"
        >
          <svg v-if="collapsed" class="collapse-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 18l6-6-6-6" />
          </svg>
          <svg v-else class="collapse-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M15 18l-6-6 6-6" />
          </svg>
        </button>
        <button v-if="!collapsed" class="run-btn" :disabled="!canRun || running" @click="$emit('run')">
          <span v-if="running">评测中</span>
          <span v-else>开始评测</span>
        </button>
      </div>
    </div>

    <div v-if="collapsed" class="collapsed-summary">
      <span class="collapsed-label">总分</span>
      <span class="collapsed-score">{{ totalScoreText }}</span>
    </div>

    <template v-else>
      <div v-if="error" class="evaluation-error">{{ error }}</div>

      <div v-if="!result" class="evaluation-empty">
        完成至少一轮学生提问和教师回复后，可以运行客观指标评测。
      </div>

      <template v-else>
        <div v-if="stale" class="evaluation-warning">
          当前聊天内容已变化，结果可能已过期。
        </div>

        <div class="score-block">
          <div class="score-value">{{ result.metrics.TotalScore.display }}</div>
          <div class="score-meta">
            <span>{{ result.modelName }}</span>
            <span>{{ evaluatedAtText }}</span>
          </div>
        </div>

        <div class="metrics-list">
          <div v-for="[key, label, metric] in metricRows" :key="key" class="metric-row">
            <div class="metric-label">{{ label }}</div>
            <div class="metric-value">
              <span>{{ metric.display }}</span>
              <span v-if="metric.rawValue !== undefined" class="metric-raw">raw {{ metric.rawValue }}</span>
            </div>
          </div>
        </div>

        <button class="annotations-toggle" @click="showAnnotations = !showAnnotations">
          {{ showAnnotations ? '收起 annotations' : '查看 annotations' }}
        </button>

        <pre v-if="showAnnotations" class="annotations-json">{{ JSON.stringify(result.annotations, null, 2) }}</pre>
      </template>
    </template>
  </aside>
</template>

<style scoped>
.evaluation-panel {
  width: 372px;
  min-width: 332px;
  border-left: 1px solid var(--border-primary);
  background: rgba(23, 23, 23, 0.86);
  padding: 18px;
  overflow-y: auto;
  backdrop-filter: blur(14px);
  transition: width var(--transition-base), min-width var(--transition-base);
}

.evaluation-panel.collapsed {
  min-width: 116px;
  width: 116px;
}

.evaluation-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}

.collapsed .evaluation-header {
  align-items: stretch;
  flex-direction: column;
}

.collapsed .evaluation-subtitle {
  display: none;
}

.evaluation-title {
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.2;
}

.evaluation-subtitle {
  color: var(--text-tertiary);
  font-size: 0.78rem;
  margin-top: 3px;
}

.collapsed .evaluation-title {
  font-size: 0.88rem;
}

.evaluation-actions {
  align-items: center;
  display: flex;
  gap: 8px;
}

.run-btn {
  border: 1px solid var(--border-secondary);
  background: var(--bg-input);
  color: var(--text-primary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.82rem;
  padding: 8px 12px;
  white-space: nowrap;
}

.collapse-btn {
  align-items: center;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-tertiary);
  cursor: pointer;
  display: flex;
  height: 32px;
  justify-content: center;
  padding: 0;
  width: 32px;
}

.collapse-btn:hover {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-primary);
}

.collapse-icon {
  height: 18px;
  width: 18px;
}

.collapsed-summary {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: 12px 10px;
}

.collapsed-label {
  color: var(--text-tertiary);
  display: block;
  font-size: 0.72rem;
  margin-bottom: 8px;
}

.collapsed-score {
  color: var(--text-primary);
  display: block;
  font-family: var(--font-mono);
  font-size: 1.15rem;
  font-weight: 700;
}

.run-btn:disabled {
  color: var(--text-tertiary);
  cursor: not-allowed;
  opacity: 0.65;
}

.run-btn:not(:disabled):hover {
  background: #f1f1f1;
  border-color: transparent;
  color: #191919;
}

.evaluation-error,
.evaluation-warning,
.evaluation-empty {
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  line-height: 1.5;
  margin-bottom: 14px;
  padding: 10px 12px;
}

.evaluation-error {
  background: rgba(243, 140, 140, 0.12);
  border: 1px solid rgba(243, 140, 140, 0.2);
  color: #ffc6c6;
}

.evaluation-warning {
  background: rgba(242, 194, 125, 0.12);
  border: 1px solid rgba(242, 194, 125, 0.22);
  color: #ffe0ad;
}

.evaluation-empty {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-secondary);
}

.score-block {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: 16px;
  margin-bottom: 14px;
  background: rgba(255, 255, 255, 0.055);
}

.score-value {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}

.score-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  color: var(--text-tertiary);
  font-size: 0.72rem;
  margin-top: 10px;
}

.metrics-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metric-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--border-primary);
  padding: 9px 0;
}

.metric-label {
  color: var(--text-secondary);
  font-size: 0.82rem;
}

.metric-value {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.84rem;
}

.metric-raw {
  color: var(--text-tertiary);
  font-family: var(--font-sans);
  font-size: 0.68rem;
}

.annotations-toggle {
  width: 100%;
  border: 1px solid var(--border-primary);
  background: transparent;
  color: var(--text-secondary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  margin-top: 14px;
  padding: 8px 10px;
}

.annotations-toggle:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-primary);
}

.annotations-json {
  margin-top: 12px;
  padding: 12px;
  border-radius: var(--radius-sm);
  background: #000000;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre-wrap;
}

@media (max-width: 980px) {
  .evaluation-panel {
    width: 100%;
    max-height: 40vh;
    border-left: none;
    border-top: 1px solid var(--border-primary);
  }
}
</style>
