<script setup lang="ts">
// 组件说明：AnnotationEditor 用于显示、选择并人工修正评测模型输出的逐轮标注。
// Props:
// - annotations: 当前会话的 ObjectiveAnnotation 列表
// - selectedIndex/selectedField: 当前选中的标注项与字段
// Emits:
// - select/update/reset/export: 用于上层组件交互
import { computed, shallowRef } from 'vue';
import {
  COGNITIVE_LEVEL_OPTIONS,
  DISCIPLINE_TRANSFER_OPTIONS,
  STUDENT_COGNITION_STATE_OPTIONS,
  TEACHER_GUIDANCE_LEVEL_OPTIONS,
  TEACHER_INTENT_OPTIONS,
  TEACHING_STRATEGY_OPTIONS,
  type AnnotationFieldKey,
  type ObjectiveAnnotation,
} from '../composables/useEvaluation';

const props = defineProps<{
  annotations: ObjectiveAnnotation[];
  selectedIndex: number;
  selectedField: AnnotationFieldKey | null;
  stale: boolean;
  canExport: boolean;
}>();

const emit = defineEmits<{
  select: [index: number, field: AnnotationFieldKey | null];
  update: [index: number, patch: Partial<ObjectiveAnnotation>];
  reset: [];
  export: [];
}>();

const showRawJson = shallowRef(false);

const selectedAnnotation = computed(() => {
  if (props.selectedIndex < 0 || props.selectedIndex >= props.annotations.length) return null;
  return props.annotations[props.selectedIndex];
});

const selectedRole = computed(() => {
  return selectedAnnotation.value?.speaker === '教师' ? 'teacher' : 'student';
});

function previewUtterance(text: string) {
  if (text.length <= 54) return text;
  return `${text.slice(0, 54)}...`;
}

function selectField(field: AnnotationFieldKey | null) {
  if (props.selectedIndex < 0) return;
  emit('select', props.selectedIndex, field);
}

function updateField(field: AnnotationFieldKey, value: string) {
  if (props.selectedIndex < 0) return;
  emit('update', props.selectedIndex, { [field]: value });
  emit('select', props.selectedIndex, field);
}

function isFieldActive(field: AnnotationFieldKey) {
  return props.selectedField === field;
}

function fieldValue(field: AnnotationFieldKey) {
  return selectedAnnotation.value?.[field] ?? '';
}

function listFieldValues(field: AnnotationFieldKey) {
  return fieldValue(field)
    .replace(/，/g, ',')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function isListOptionSelected(field: AnnotationFieldKey, option: string) {
  return listFieldValues(field).includes(option);
}

function toggleListOption(field: AnnotationFieldKey, option: string) {
  const values = listFieldValues(field);
  const nextValues = values.includes(option)
    ? values.filter((value) => value !== option)
    : [...values, option];
  updateField(field, nextValues.join(','));
}
</script>

<template>
  <section class="annotation-editor">
    <div class="editor-toolbar">
      <button class="toolbar-btn" :disabled="!canExport" @click="$emit('export')">导出 JSONL</button>
      <button class="toolbar-btn" :disabled="annotations.length === 0" @click="$emit('reset')">恢复 AI 标注</button>
      <button class="toolbar-btn" :disabled="annotations.length === 0" @click="showRawJson = !showRawJson">
        {{ showRawJson ? '收起 JSON' : '查看 JSON' }}
      </button>
    </div>

    <div v-if="stale" class="editor-hint">
      当前 annotation 对应的聊天轨迹已变化，建议重新评测后再导出。
    </div>

    <div v-if="annotations.length === 0" class="editor-empty">
      运行客观评测后，这里会显示逐轮 annotation，可直接人工修订。
    </div>

    <template v-else>
      <div class="annotation-list">
        <button v-for="(annotation, index) in annotations" :key="`${annotation.speaker}-${index}`"
          class="annotation-card" :class="{ selected: index === selectedIndex }" @click="$emit('select', index, null)">
          <span class="annotation-card-head">
            <span class="speaker-badge" :class="annotation.speaker === '教师' ? 'speaker-teacher' : 'speaker-student'">
              {{ annotation.speaker }}
            </span>
            <span class="turn-index">#{{ index + 1 }}</span>
          </span>
          <span class="annotation-card-text">{{ previewUtterance(annotation.utterance) }}</span>
        </button>
      </div>

      <div v-if="selectedAnnotation" class="editor-detail">
        <div class="detail-section">
          <div class="section-title">当前轮次</div>
          <div class="readonly-card">
            <div class="readonly-row">
              <span class="readonly-label">发言者</span>
              <span class="readonly-value">{{ selectedAnnotation.speaker }}</span>
            </div>
            <div class="readonly-row utterance-row">
              <span class="readonly-label">原始发言</span>
              <div class="utterance-box">{{ selectedAnnotation.utterance }}</div>
            </div>
          </div>
        </div>

        <div class="detail-section">
          <div class="section-title">通用字段</div>

          <label class="field-block" :class="{ active: isFieldActive('discipline') }">
            <span class="field-label">学科</span>
            <input class="field-input" type="text" :value="fieldValue('discipline')" placeholder="如：生物,地质学"
              @focus="selectField('discipline')"
              @input="updateField('discipline', ($event.target as HTMLInputElement).value)" />
          </label>

          <div class="field-block" :class="{ active: isFieldActive('discipline_transfer') }">
            <span class="field-label">跨学科迁移</span>
            <div class="option-group">
              <button v-for="option in DISCIPLINE_TRANSFER_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: fieldValue('discipline_transfer') === option }"
                @click="updateField('discipline_transfer', option)">
                {{ option }}
              </button>
            </div>
          </div>

          <div class="field-block" :class="{ active: isFieldActive('cognitive_level') }">
            <span class="field-label">Bloom</span>
            <div class="option-group">
              <button v-for="option in COGNITIVE_LEVEL_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: fieldValue('cognitive_level') === option }"
                @click="updateField('cognitive_level', option)">
                {{ option }}
              </button>
            </div>
          </div>
        </div>

        <div v-if="selectedRole === 'teacher'" class="detail-section">
          <div class="section-title">教师字段</div>

          <div class="field-block" :class="{ active: isFieldActive('teacher_intent') }">
            <span class="field-label">教学意图</span>
            <div class="option-group">
              <button v-for="option in TEACHER_INTENT_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: fieldValue('teacher_intent') === option }"
                @click="updateField('teacher_intent', option)">
                {{ option }}
              </button>
            </div>
          </div>

          <div class="field-block" :class="{ active: isFieldActive('teaching_strategy') }">
            <span class="field-label">教学策略</span>
            <div class="option-group">
              <button v-for="option in TEACHING_STRATEGY_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: isListOptionSelected('teaching_strategy', option) }"
                @click="toggleListOption('teaching_strategy', option)">
                {{ option }}
              </button>
            </div>
          </div>

          <div class="field-block" :class="{ active: isFieldActive('teacher_guidance_level') }">
            <span class="field-label">引导等级</span>
            <div class="option-group">
              <button v-for="option in TEACHER_GUIDANCE_LEVEL_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: fieldValue('teacher_guidance_level') === option }"
                @click="updateField('teacher_guidance_level', option)">
                {{ option }}
              </button>
            </div>
          </div>
        </div>

        <div v-else class="detail-section">
          <div class="section-title">学生字段</div>

          <div class="field-block" :class="{ active: isFieldActive('student_cognition_state') }">
            <span class="field-label">认知状态</span>
            <div class="option-group">
              <button v-for="option in STUDENT_COGNITION_STATE_OPTIONS" :key="option" class="option-chip"
                :class="{ selected: fieldValue('student_cognition_state') === option }"
                @click="updateField('student_cognition_state', option)">
                {{ option }}
              </button>
            </div>
          </div>
        </div>

        <pre v-if="showRawJson" class="raw-json">{{ JSON.stringify(annotations, null, 2) }}</pre>
      </div>
    </template>
  </section>
</template>

<style scoped>
.annotation-editor {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}

.editor-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.toolbar-btn {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.76rem;
  padding: 8px 10px;
}

.toolbar-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-primary);
}

.toolbar-btn:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.editor-hint,
.editor-empty {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.78rem;
  line-height: 1.5;
  padding: 10px 12px;
}

.annotation-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 220px;
  overflow-y: auto;
  padding-right: 2px;
}

.annotation-card {
  align-items: flex-start;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 10px 11px;
  text-align: left;
}

.annotation-card.selected {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.18);
}

.annotation-card-head {
  align-items: center;
  display: flex;
  gap: 8px;
  width: 100%;
}

.speaker-badge {
  border-radius: var(--radius-full);
  font-size: 0.67rem;
  font-weight: 700;
  padding: 3px 8px;
}

.speaker-teacher {
  background: rgba(141, 206, 255, 0.14);
  color: #c7e8ff;
}

.speaker-student {
  background: rgba(255, 205, 149, 0.14);
  color: #ffd8a4;
}

.turn-index {
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 0.7rem;
}

.annotation-card-text {
  color: var(--text-primary);
  font-size: 0.8rem;
  line-height: 1.45;
}

.editor-detail {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.detail-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.section-title {
  color: var(--text-primary);
  font-size: 0.78rem;
  font-weight: 650;
}

.readonly-card,
.field-block {
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 10px 11px;
}

.field-block.active {
  border-color: rgba(255, 255, 255, 0.18);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}

.readonly-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.utterance-row {
  margin-top: 10px;
}

.readonly-label,
.field-label {
  color: var(--text-tertiary);
  font-size: 0.71rem;
}

.readonly-value {
  color: var(--text-primary);
  font-size: 0.82rem;
}

.utterance-box {
  color: var(--text-primary);
  font-size: 0.82rem;
  line-height: 1.55;
  white-space: pre-wrap;
}

.field-input {
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 0.82rem;
  margin-top: 8px;
  padding: 9px 10px;
  width: 100%;
}

.option-group {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 8px;
}

.option-chip {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.75rem;
  padding: 6px 9px;
}

.option-chip.selected {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(255, 255, 255, 0.22);
  color: var(--text-primary);
}

.raw-json {
  background: #000000;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  line-height: 1.5;
  margin: 0;
  overflow-x: auto;
  padding: 12px;
  white-space: pre-wrap;
}
</style>
