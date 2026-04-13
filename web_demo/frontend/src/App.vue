<template>
  <div class="layout-wrapper">
    <Sidebar 
      :sessions="activeModeSessions" 
      :current-id="currentSessionId" 
      @new="createNewSession"
      @switch="switchSession"
      @delete="deleteSession"
      @open-settings="showGlobalSettings = true"
    />
    
    <div class="app">
      <!-- Top Bar -->
      <header class="top-bar">
        <div class="logo">
          <span class="logo-icon">⚡</span>
          <h1>Teaching Arena</h1>
          <span class="logo-badge">{{ chatMode === 'sbs' ? 'Side-by-Side' : 'Single Model' }}</span>
        </div>
        <div class="top-actions">
          <div class="mode-switch">
            <button 
              class="mode-btn" 
              :class="{ active: chatMode === 'single' }"
              @click="setChatMode('single')"
            >
              单模型
            </button>
            <button 
              class="mode-btn" 
              :class="{ active: chatMode === 'sbs' }"
              @click="setChatMode('sbs')"
            >
              Side-by-Side
            </button>
          </div>
          <!-- Clear Chat is replaced by deleting session or making a new one, but we can keep it to clear current messages -->
          <button class="action-btn" @click="clearChat" title="清除当前内容">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            重置
          </button>
        </div>
      </header>

      <!-- Main Content: Side-by-Side Panels -->
      <main class="main-content">
        <div class="panels-container">
          <ChatPanel
            side="a"
            :chat-mode="chatMode"
            :model-name="globalModelConfig.panelA.name"
            :prompt-name="panelA.promptName"
            :messages="messages"
            :responses="panelA.responses"
            :streaming="panelA.streaming"
            :streaming-text="panelA.streamingText"
            :error="panelA.error"
            @open-prompt="showPromptA = true"
          />
          <div class="panel-divider" v-if="chatMode === 'sbs'"></div>
          <ChatPanel
            v-if="chatMode === 'sbs'"
            side="b"
            :chat-mode="chatMode"
            :model-name="globalModelConfig.panelB.name"
            :prompt-name="panelB.promptName"
            :messages="messages"
            :responses="panelB.responses"
            :streaming="panelB.streaming"
            :streaming-text="panelB.streamingText"
            :error="panelB.error"
            @open-prompt="showPromptB = true"
          />
        </div>
      </main>

      <!-- Input Bar -->
      <InputBar
        :is-streaming="panelA.streaming || panelB.streaming"
        @send="handleSend"
        @stop="stopStreaming"
      />

      <!-- Modals -->
      <GlobalSettings 
        :visible="showGlobalSettings" 
        :config="globalModelConfig"
        @close="showGlobalSettings = false" 
        @save="saveGlobalConfig" 
      />
      
      <PromptEditor
        :visible="showPromptA" side="a" :current-prompt="panelA.systemPrompt"
        @close="showPromptA = false" @save="savePromptA"
      />
      <PromptEditor
        :visible="showPromptB" side="b" :current-prompt="panelB.systemPrompt"
        @close="showPromptB = false" @save="savePromptB"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useChat } from './composables/useChat'
import ChatPanel from './components/ChatPanel.vue'
import InputBar from './components/InputBar.vue'
import GlobalSettings from './components/GlobalSettings.vue'
import PromptEditor from './components/PromptEditor.vue'
import Sidebar from './components/Sidebar.vue'

const { 
  messages, panelA, panelB, sendMessage, stopStreaming, clearChat,
  globalModelConfig,
  chatMode, setChatMode,
  sessions, activeModeSessions, currentSessionId, createNewSession, switchSession, deleteSession
} = useChat()

const showGlobalSettings = ref(false)
const showPromptA = ref(false)
const showPromptB = ref(false)

function handleSend(content: string) {
  sendMessage(content, chatMode.value === 'sbs' ? ['a', 'b'] : ['a'])
}

function saveGlobalConfig(newConfig: typeof globalModelConfig) {
  globalModelConfig.panelA = { ...newConfig.panelA }
  globalModelConfig.panelB = { ...newConfig.panelB }
}

function savePromptA({ content, name }: { content: string; name: string }) {
  panelA.systemPrompt = content
  panelA.promptName = name
}

function savePromptB({ content, name }: { content: string; name: string }) {
  panelB.systemPrompt = content
  panelB.promptName = name
}
</script>

<style scoped>
.layout-wrapper {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}

.app { display: flex; flex-direction: column; flex: 1; min-width: 0; position: relative; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 24px; background: var(--bg-primary); 
  flex-shrink: 0; z-index: 10;
}
.logo { display: flex; align-items: center; gap: 8px; cursor: default; }
.logo-icon { font-size: 1.25rem; font-weight: 500; }
.logo h1 {
  font-size: 1.1rem; font-weight: 600;
  color: var(--text-primary);
}
.logo-badge {
  padding: 2px 6px; border-radius: 4px;
  background: var(--bg-input); 
  color: var(--text-secondary); font-size: 0.65rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  margin-left: 4px;
}
.top-actions { display: flex; gap: 8px; }
.action-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: var(--radius-sm);
  background: transparent; border: none;
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.85rem; cursor: pointer; transition: all var(--transition-fast);
}
.action-btn:hover {
  color: var(--text-primary);
  background: var(--bg-input);
}

.mode-switch {
  display: flex;
  background: var(--bg-input);
  padding: 4px;
  border-radius: var(--radius-md);
  margin-right: 12px;
}
.mode-btn {
  padding: 6px 12px;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.85rem;
  font-family: var(--font-sans);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.mode-btn:hover {
  color: var(--text-primary);
}
.mode-btn.active {
  background: var(--border-primary);
  color: var(--text-primary);
  font-weight: 500;
  box-shadow: var(--shadow-sm);
}

.main-content { flex: 1; overflow: hidden; }
.panels-container { display: flex; gap: 0; height: 100%; }
.panels-container > :first-child,
.panels-container > :last-child { flex: 1; min-width: 0; }
.panel-divider { width: 1px; background: var(--bg-divider); flex-shrink: 0; }

@media (max-width: 768px) {
  .panels-container { flex-direction: column; gap: 8px; }
  .panel-divider { width: 100%; height: 1px; margin: 0; }
}
</style>
