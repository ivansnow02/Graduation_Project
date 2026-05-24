<template>
  <div class="chat-panel" :class="[`panel-${side}`, { 'is-streaming': streaming }]">
    <div class="panel-header">
      <div class="panel-label">
        <span class="panel-dot" :class="`dot-${side}`"></span>
        <span class="model-name">{{ modelName || '未配置模型' }}</span>
        <span class="side-badge" v-if="chatMode === 'sbs'">{{ side === 'a' ? '基准模型' : '对照模型' }}</span>
      </div>
      <div class="panel-actions">
        <button v-if="promptName" class="action-text-btn" @click="$emit('openPrompt')">
          {{ promptName }}
        </button>
        <button v-else class="icon-btn" title="设置 Prompt" @click="$emit('openPrompt')">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
          </svg>
        </button>
      </div>
    </div>

    <div ref="messagesArea" class="messages-area">
      <div v-if="displayMessages.length === 0 && !streaming" class="empty-state">
        <div class="empty-logo">评测</div>
        <h2>开始一次教学对话</h2>
        <p>配置模型与系统提示词后，输入学生问题以观察模型的教学引导行为。</p>
      </div>

      <div class="message-feed">
        <div v-for="(msg, idx) in displayMessages" :key="idx" class="message-row" :class="`role-${msg.role}`">
          <div v-if="msg.role === 'assistant'" class="avatar assistant-avatar" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a10 10 0 1 0 10 10H12z" />
              <path d="M12 2a10 10 0 0 0 0 20V2z" />
            </svg>
          </div>
          <div class="message-content">
            <div class="message-bubble">
              <div class="markdown-body" v-html="renderMarkdown(msg.content, msg.completed)"></div>
              <div v-if="msg.completed" class="completion-badge">
                <svg class="completion-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
                已完成
              </div>
            </div>
            <div v-if="msg.annotationFields.length > 0" class="annotation-row"
              :class="{ selected: msg.annotationIndex === selectedAnnotationIndex }" aria-label="客观评测 annotation">
              <button v-for="field in msg.annotationFields" :key="`${field.key}-${field.label}`" class="annotation-chip"
                :class="{ selected: msg.annotationIndex === selectedAnnotationIndex && selectedAnnotationField === field.key }"
                @click="$emit('selectAnnotation', msg.annotationIndex, field.key)">
                <span class="annotation-label">{{ field.label }}</span>
                <span class="annotation-value">{{ field.value }}</span>
              </button>
            </div>
          </div>
        </div>

        <div v-if="streaming" class="message-row role-assistant">
          <div class="avatar assistant-avatar" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a10 10 0 1 0 10 10H12z" />
              <path d="M12 2a10 10 0 0 0 0 20V2z" />
            </svg>
          </div>
          <div class="message-content">
            <div class="message-bubble">
              <div v-if="streamingText" class="markdown-body"
                v-html="renderMarkdown(streamingText, isCompleted(streamingText))"></div>
              <div v-else class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="error" class="error-banner">
        <span>出现错误：{{ error }}</span>
      </div>
      <div ref="bottomAnchor" class="bottom-anchor"></div>
      <div class="bottom-spacer"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 组件说明：ChatPanel 展示一侧模型的聊天历史、流式生成与客观评测注释快捷查看。
// 重要职责：渲染消息、显示流式生成状态、暴露选择 annotation 的事件。
import { computed, watch, nextTick, ref, onMounted } from 'vue';
import { marked } from 'marked';
import type { AnnotationFieldKey, ObjectiveAnnotation } from '../composables/useEvaluation';

type MessageRole = 'user' | 'assistant';

interface AnnotationField {
  key: AnnotationFieldKey;
  label: string;
  value: string;
}

interface DisplayMessage {
  role: MessageRole;
  content: string;
  completed?: boolean;
  annotationIndex: number;
  annotationFields: AnnotationField[];
}

const props = defineProps<{
  side: string;
  chatMode?: string;
  modelName: string;
  promptName: string;
  messages: { role: string; content: string; }[];
  responses: Record<number, string>;
  streaming: boolean;
  streamingText: string;
  error: string;
  annotations?: ObjectiveAnnotation[];
  selectedAnnotationIndex?: number;
  selectedAnnotationField?: AnnotationFieldKey | null;
}>();

defineEmits<{
  openPrompt: [];
  openModel: [];
  selectAnnotation: [index: number, field: AnnotationFieldKey];
}>();

const messagesArea = ref<HTMLElement | null>(null);
const bottomAnchor = ref<HTMLElement | null>(null);

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text: string, completed = false): string {
  if (!text) return '';
  // 使用 marked 渲染 markdown，同时先处理自定义的结束标记逻辑
  return marked.parse(toDisplayContent(text, completed)) as string;
}

function stripCompletionMarker(text: string) {
  return text.replace(/\s*\[结束\]\s*/g, '').trim();
}

function toDisplayContent(text: string, completed: boolean) {
  const content = stripCompletionMarker(text);
  if (content) return content;
  if (completed) return '本轮教学目标已完成。';
  return '';
}

function isCompleted(text: string) {
  return /\[结束\]/.test(text);
}

function compactField(key: AnnotationFieldKey, label: string, value?: string): AnnotationField | null {
  const normalized = value?.trim();
  if (!normalized) return null;
  return { key, label, value: normalized };
}

function getAnnotationFields(role: MessageRole, annotation?: ObjectiveAnnotation): AnnotationField[] {
  if (!annotation) return [];
  const fields =
    role === 'user'
      ? [
        compactField('student_cognition_state', '认知', annotation.student_cognition_state),
        compactField('cognitive_level', 'Bloom', annotation.cognitive_level),
        compactField('discipline', '学科', annotation.discipline),
        compactField('discipline_transfer', '迁移', annotation.discipline_transfer),
      ]
      : [
        compactField('teacher_intent', '意图', annotation.teacher_intent),
        compactField('teaching_strategy', '策略', annotation.teaching_strategy),
        compactField('teacher_guidance_level', '引导', annotation.teacher_guidance_level),
        compactField('discipline', '学科', annotation.discipline),
        compactField('discipline_transfer', '迁移', annotation.discipline_transfer),
        compactField('cognitive_level', 'Bloom', annotation.cognitive_level),
      ];
  return fields.filter((field): field is AnnotationField => Boolean(field));
}

const displayMessages = computed(() => {
  const result: DisplayMessage[] = [];
  const annotations = props.annotations ?? [];
  let annotationIndex = 0;
  for (let i = 0; i < props.messages.length; i++) {
    const msg = props.messages[i];
    if (msg.role === 'user') {
      const response = props.responses[i];
      const hasEvaluatedTurn = Boolean(msg.content.trim() && response?.trim());
      const studentAnnotation = hasEvaluatedTurn ? annotations[annotationIndex] : undefined;
      const teacherAnnotation = hasEvaluatedTurn ? annotations[annotationIndex + 1] : undefined;
      result.push({
        role: 'user',
        content: msg.content,
        annotationIndex: hasEvaluatedTurn ? annotationIndex : -1,
        annotationFields: getAnnotationFields('user', studentAnnotation),
      });
      if (response !== undefined) {
        result.push({
          role: 'assistant',
          content: response,
          completed: isCompleted(response),
          annotationIndex: hasEvaluatedTurn ? annotationIndex + 1 : -1,
          annotationFields: getAnnotationFields('assistant', teacherAnnotation),
        });
        if (hasEvaluatedTurn) annotationIndex += 2;
      }
    }
  }
  return result;
});

// displayMessages 负责将原始 messages/responses 与 annotations 对齐，
// 输出适合渲染的扁平消息列表：每个学生消息后紧跟助手回复（若存在），
// 并为每条消息绑定对应的 annotation 索引与快速展示字段。

function scrollToBottom() {
  const area = messagesArea.value;
  if (!area) return;
  area.scrollTop = area.scrollHeight;
  bottomAnchor.value?.scrollIntoView({ block: 'end' });
}

function scheduleScrollToBottom() {
  nextTick(() => {
    scrollToBottom();
    requestAnimationFrame(() => {
      scrollToBottom();
      window.setTimeout(scrollToBottom, 0);
    });
  });
}

watch(
  [() => props.streamingText, () => props.streaming, () => displayMessages.value.length],
  scheduleScrollToBottom,
  { flush: 'post' }
);

onMounted(scheduleScrollToBottom);
</script>

<style scoped>
.chat-panel {
  background: transparent;
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  min-width: 0;
  position: relative;
}

.panel-header {
  align-items: center;
  background: rgba(31, 31, 31, 0.74);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  flex-shrink: 0;
  justify-content: space-between;
  min-height: 54px;
  padding: 10px 22px;
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(16px);
}

.panel-label {
  align-items: center;
  display: flex;
  gap: 9px;
  min-width: 0;
}

.panel-dot {
  border-radius: var(--radius-full);
  flex: 0 0 auto;
  height: 8px;
  width: 8px;
}

.dot-a {
  background: var(--accent-cyan);
}

.dot-b {
  background: var(--accent-success);
}

.model-name {
  color: var(--text-primary);
  font-size: 0.92rem;
  font-weight: 650;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.side-badge {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-tertiary);
  font-size: 0.68rem;
  font-weight: 650;
  padding: 3px 7px;
  text-transform: uppercase;
  white-space: nowrap;
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-text-btn {
  background: transparent;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.8rem;
  max-width: 220px;
  overflow: hidden;
  padding: 5px 12px;
  text-overflow: ellipsis;
  transition: all var(--transition-fast);
  white-space: nowrap;
}

.action-text-btn:hover {
  background: var(--bg-input);
  color: var(--text-primary);
}

.icon-btn {
  align-items: center;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-tertiary);
  cursor: pointer;
  display: flex;
  height: 32px;
  justify-content: center;
  transition: all var(--transition-fast);
  width: 32px;
}

.icon-btn:hover {
  background: var(--bg-input);
  color: var(--text-primary);
}

.messages-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 0 var(--composer-safe-area, 220px);
  scroll-behavior: auto;
  scroll-padding-bottom: var(--composer-safe-area, 220px);
}

.message-feed {
  margin: 0 auto;
  max-width: 820px;
  padding: 0 28px;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.empty-state {
  align-items: center;
  color: var(--text-tertiary);
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  margin: auto;
  max-width: 520px;
  padding: 32px;
  text-align: center;
}

.empty-logo {
  align-items: center;
  background: var(--gradient-primary);
  border-radius: var(--radius-lg);
  color: #171717;
  display: flex;
  font-size: 0.96rem;
  font-weight: 800;
  height: 48px;
  justify-content: center;
  margin-bottom: 16px;
  width: 56px;
}

.empty-state h2 {
  color: var(--text-primary);
  font-size: 1.32rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.empty-state p {
  color: var(--text-tertiary);
  font-size: 0.94rem;
  line-height: 1.7;
}

.message-row {
  animation: fadeIn 0.22s ease;
  display: flex;
  margin-bottom: 24px;
  width: 100%;
}

.message-content {
  align-items: flex-start;
  display: flex;
  flex-direction: column;
  max-width: 100%;
  min-width: 0;
}

.role-user {
  justify-content: flex-end;
}

.role-user .message-content {
  align-items: flex-end;
  max-width: min(78%, 680px);
}

.role-assistant .message-content {
  flex: 1;
}

.role-assistant {
  justify-content: flex-start;
}

.role-user .message-bubble {
  background: var(--bg-user-bubble);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 20px;
  color: var(--text-primary);
  font-size: 1rem;
  max-width: 100%;
  padding: 11px 17px;
}

.role-assistant .message-bubble {
  background: var(--bg-assistant);
  max-width: 100%;
  min-width: 0;
  padding: 2px 0;
}

.completion-badge {
  align-items: center;
  border: 1px solid rgba(143, 211, 176, 0.32);
  border-radius: var(--radius-full);
  color: var(--accent-success);
  display: inline-flex;
  font-size: 0.78rem;
  gap: 5px;
  margin-top: 10px;
  padding: 3px 9px;
}

.completion-icon {
  height: 13px;
  width: 13px;
}

.annotation-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
  max-width: 100%;
}

.role-user .annotation-row {
  justify-content: flex-end;
}

.annotation-row.selected {
  filter: brightness(1.04);
}

.annotation-chip {
  align-items: center;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  cursor: pointer;
  display: inline-flex;
  font-size: 0.72rem;
  gap: 5px;
  line-height: 1.3;
  max-width: 100%;
  padding: 4px 8px;
  transition: background var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast);
}

.annotation-chip:hover {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-primary);
}

.annotation-chip.selected {
  background: rgba(255, 255, 255, 0.13);
  border-color: rgba(255, 255, 255, 0.22);
}

.annotation-label {
  color: var(--text-tertiary);
  flex: 0 0 auto;
}

.annotation-value {
  color: var(--text-primary);
  min-width: 0;
  overflow-wrap: anywhere;
}

.avatar {
  align-items: center;
  border: 1px solid var(--border-primary);
  border-radius: 50%;
  display: flex;
  flex-shrink: 0;
  height: 30px;
  justify-content: center;
  margin-right: 16px;
  width: 30px;
}

.assistant-avatar {
  background: #f4f4f4;
  border: 0;
  color: #171717;
}

.error-banner {
  background: rgba(243, 140, 140, 0.12);
  border: 1px solid rgba(243, 140, 140, 0.2);
  border-radius: var(--radius-md);
  color: #ffc6c6;
  margin: 16px auto;
  max-width: 800px;
  padding: 12px 16px;
  font-size: 0.9rem;
}

.bottom-spacer {
  flex: 0 0 var(--composer-safe-area, 220px);
}

.bottom-anchor {
  height: 1px;
}

@media (max-width: 768px) {
  .panel-header {
    padding: 9px 14px;
  }

  .message-feed {
    padding: 0 16px;
  }

  .role-user .message-content {
    max-width: 90%;
  }

  .action-text-btn {
    max-width: 140px;
  }
}
</style>
