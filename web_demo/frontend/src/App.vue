<template>
  <div class="layout-wrapper">
    <Sidebar v-if="workspaceMode === 'chat'" :sessions="activeModeSessions" :current-id="currentSessionId" @new="createNewSession"
      @switch="switchSession" @delete="deleteSession" @open-settings="showGlobalSettings = true" />

    <div class="app">
      <header class="top-bar">
        <div class="logo">
          <span class="logo-icon">评测</span>
          <div class="logo-copy">
            <h1>基于主动协作机制的苏格拉底式教学对话评测系统</h1>
            <p>面向苏格拉底式教学对话模型的客观评测与指标分析环境</p>
          </div>
          <span class="logo-badge">{{ workspaceBadge }}</span>
        </div>
        <div class="top-actions">
          <div class="mode-switch">
            <button class="mode-btn" :class="{ active: workspaceMode === 'chat' && chatMode === 'single' }"
              @click="openChatMode('single')">
              单模型评测
            </button>
            <button class="mode-btn" :class="{ active: workspaceMode === 'chat' && chatMode === 'sbs' }"
              @click="openChatMode('sbs')">
              双模型对照
            </button>
            <button class="mode-btn" :class="{ active: workspaceMode === 'metrics' }" @click="workspaceMode = 'metrics'">
              指标对比
            </button>
          </div>
          <button v-if="workspaceMode === 'chat'" class="action-btn" @click="clearChat" title="清除当前内容">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            重置
          </button>
        </div>
      </header>

      <main class="main-content" :class="{ 'metrics-content': workspaceMode === 'metrics' }">
        <MetricsDashboard v-if="workspaceMode === 'metrics'" />
        <template v-else>
          <div class="chat-workspace">
            <div class="panels-container">
              <ChatPanel side="a" :chat-mode="chatMode" :model-name="globalModelConfig.panelA.name"
                :prompt-name="panelA.promptName" :messages="messages" :responses="panelA.responses"
                :streaming="panelA.streaming" :streaming-text="panelA.streamingText" :error="panelA.error"
                :annotations="chatMode === 'single' && !evaluationStale ? evaluation.result?.annotations ?? [] : []"
                :selected-annotation-index="selectedAnnotation.index"
                :selected-annotation-field="selectedAnnotation.field" @open-prompt="showPromptA = true"
                @select-annotation="handleSelectAnnotation" />
              <div class="panel-divider" v-if="chatMode === 'sbs'"></div>
              <ChatPanel v-if="chatMode === 'sbs'" side="b" :chat-mode="chatMode"
                :model-name="globalModelConfig.panelB.name" :prompt-name="panelB.promptName" :messages="messages"
                :responses="panelB.responses" :streaming="panelB.streaming" :streaming-text="panelB.streamingText"
                :error="panelB.error" @open-prompt="showPromptB = true" />
            </div>
            <InputBar :is-streaming="panelA.streaming || panelB.streaming" @send="handleSend" @stop="stopStreaming" />
          </div>
          <EvaluationPanel v-if="chatMode === 'single'" :collapsed="evaluationCollapsed" :result="evaluation.result"
            :running="evaluation.running" :error="evaluation.error" :stale="evaluationStale" :can-run="canRunEvaluation"
            :can-export="canExportEvaluation" :selected-index="selectedAnnotation.index"
            :selected-field="selectedAnnotation.field" @toggle-collapse="evaluationCollapsed = !evaluationCollapsed"
            @run="runObjectiveEvaluation" @select-annotation="selectAnnotation" @update-annotation="updateAnnotation"
            @reset-annotations="resetEditedAnnotations" @export-annotations="exportEvaluationJsonl" />
        </template>
      </main>

      <GlobalSettings :visible="showGlobalSettings" :config="globalModelConfig" @close="showGlobalSettings = false"
        @save="saveGlobalConfig" />

      <PromptEditor :visible="showPromptA" side="a" :current-prompt="panelA.systemPrompt" @close="showPromptA = false"
        @save="savePromptA" />
      <PromptEditor :visible="showPromptB" side="b" :current-prompt="panelB.systemPrompt" @close="showPromptB = false"
        @save="savePromptB" />
    </div>
  </div>
</template>

<script setup lang="ts">
// 应用入口：组合 `useChat` 提供的状态与操作，挂载主要布局和面板组件
// 该文件主要负责连接业务逻辑与 UI 组件，不包含复杂逻辑实现
import { computed, ref } from 'vue';
import { useChat } from './composables/useChat';
import type { ChatMode } from './composables/useChat';
import type { AnnotationFieldKey } from './composables/useEvaluation';
import ChatPanel from './components/ChatPanel.vue';
import InputBar from './components/InputBar.vue';
import GlobalSettings from './components/GlobalSettings.vue';
import PromptEditor from './components/PromptEditor.vue';
import Sidebar from './components/Sidebar.vue';
import EvaluationPanel from './components/EvaluationPanel.vue';
import MetricsDashboard from './components/MetricsDashboard.vue';

const {
  messages, panelA, panelB, sendMessage, stopStreaming, clearChat,
  globalModelConfig,
  evaluation, evaluationStale, canRunEvaluation, canExportEvaluation, runObjectiveEvaluation,
  selectedAnnotation, selectAnnotation, updateAnnotation, resetEditedAnnotations, exportEvaluationJsonl,
  chatMode, setChatMode,
  sessions, activeModeSessions, currentSessionId, createNewSession, switchSession, deleteSession
} = useChat();

const showGlobalSettings = ref(false);
const showPromptA = ref(false);
const showPromptB = ref(false);
const evaluationCollapsed = ref(false);
const workspaceMode = ref<'chat' | 'metrics'>('chat');
const workspaceBadge = computed(() => {
  if (workspaceMode.value === 'metrics') return '指标对比';
  return chatMode.value === 'sbs' ? '双模型对照' : '单模型评测';
});

function openChatMode(mode: ChatMode) {
  workspaceMode.value = 'chat';
  setChatMode(mode);
}

function handleSend(content: string) {
  sendMessage(content, chatMode.value === 'sbs' ? ['a', 'b'] : ['a']);
}

function handleSelectAnnotation(index: number, field: AnnotationFieldKey) {
  evaluationCollapsed.value = false;
  selectAnnotation(index, field);
}

function saveGlobalConfig(newConfig: typeof globalModelConfig) {
  globalModelConfig.panelA = { ...newConfig.panelA };
  globalModelConfig.panelB = { ...newConfig.panelB };
  globalModelConfig.evaluator = { ...newConfig.evaluator };
  globalModelConfig.modelProfiles = newConfig.modelProfiles.map(profile => ({ ...profile }));
  globalModelConfig.panelAProfileId = newConfig.panelAProfileId;
  globalModelConfig.panelBProfileId = newConfig.panelBProfileId;
  globalModelConfig.evaluatorProfileId = newConfig.evaluatorProfileId;
  globalModelConfig.generation = { ...newConfig.generation };
}

function savePromptA({ content, name }: { content: string; name: string; }) {
  panelA.systemPrompt = content;
  panelA.promptName = name;
}

function savePromptB({ content, name }: { content: string; name: string; }) {
  panelB.systemPrompt = content;
  panelB.promptName = name;
}
</script>

<style scoped>
.layout-wrapper {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: var(--bg-primary);
}

.app {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  position: relative;
  background:
    radial-gradient(circle at 50% -18%, rgba(255, 255, 255, 0.075), transparent 28%),
    var(--bg-primary);
}

.top-bar {
  align-items: center;
  background: rgba(31, 31, 31, 0.84);
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  flex-shrink: 0;
  justify-content: space-between;
  min-height: 66px;
  padding: 12px 24px;
  z-index: 20;
  backdrop-filter: blur(18px);
}

.logo {
  align-items: center;
  cursor: default;
  display: flex;
  gap: 12px;
  min-width: 0;
}

.logo-icon {
  align-items: center;
  background: var(--gradient-primary);
  border-radius: var(--radius-sm);
  color: #191919;
  display: flex;
  flex: 0 0 auto;
  font-size: 0.78rem;
  font-weight: 800;
  height: 34px;
  justify-content: center;
  letter-spacing: 0;
  width: 42px;
}

.logo-copy {
  flex: 1 1 auto;
  min-width: 0;
}

.logo h1 {
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 650;
  line-height: 1.2;
  max-width: min(46vw, 520px);
  overflow-wrap: anywhere;
}

.logo-copy p {
  color: var(--text-tertiary);
  font-size: 0.76rem;
  line-height: 1.3;
  margin-top: 2px;
}

.logo-badge {
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  font-size: 0.66rem;
  font-weight: 650;
  letter-spacing: 0;
  margin-left: 4px;
  padding: 4px 8px;
  text-transform: uppercase;
  white-space: nowrap;
}

.top-actions {
  align-items: center;
  display: flex;
  gap: 10px;
}

.action-btn {
  align-items: center;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  font-family: var(--font-sans);
  font-size: 0.84rem;
  gap: 7px;
  min-height: 36px;
  padding: 7px 12px;
  transition: all var(--transition-fast);
}

.action-btn:hover {
  background: var(--bg-input);
  border-color: var(--border-primary);
  color: var(--text-primary);
}

.mode-switch {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  display: flex;
  gap: 2px;
  padding: 3px;
}

.mode-btn {
  background: transparent;
  border: none;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.84rem;
  min-height: 32px;
  padding: 6px 12px;
  transition: all var(--transition-fast);
  white-space: nowrap;
}

.mode-btn:hover {
  color: var(--text-primary);
}

.mode-btn.active {
  background: #f2f2f2;
  box-shadow: var(--shadow-sm);
  color: #1d1d1d;
  font-weight: 650;
}

.main-content {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.chat-workspace {
  --composer-safe-area: 220px;
  display: flex;
  flex: 1;
  min-height: 0;
  min-width: 0;
  position: relative;
}

.panels-container {
  display: flex;
  flex: 1;
  gap: 0;
  height: 100%;
  min-height: 0;
  min-width: 0;
}

.panels-container> :first-child,
.panels-container> :last-child {
  flex: 1;
  min-width: 0;
}

.panel-divider {
  background: var(--bg-divider);
  flex-shrink: 0;
  width: 1px;
}

@media (max-width: 920px) {
  .top-bar {
    align-items: flex-start;
    flex-direction: column;
    gap: 12px;
    padding: 12px 16px;
  }

  .top-actions {
    width: 100%;
  }

  .mode-switch {
    flex: 1;
  }

  .mode-btn {
    flex: 1;
  }

  .action-btn {
    padding-inline: 10px;
  }
}

@media (max-width: 768px) {

  .logo-copy p,
  .logo-badge {
    display: none;
  }

  .main-content {
    flex-direction: column;
  }

  .panels-container {
    flex-direction: column;
  }

  .chat-workspace {
    min-height: 0;
    --composer-safe-area: 200px;
  }

  .panel-divider {
    height: 1px;
    width: 100%;
  }
}
</style>
