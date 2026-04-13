<template>
  <div class="chat-panel" :class="[`panel-${side}`, { 'is-streaming': streaming }]">
    <!-- Clean Minimalist Header -->
    <div class="panel-header">
      <div class="panel-label">
        <span class="model-name">{{ modelName || '未配置模型' }}</span>
        <span class="side-badge" v-if="chatMode === 'sbs'">Model {{ side.toUpperCase() }}</span>
      </div>
      <div class="panel-actions">
        <button v-if="promptName" class="action-text-btn" @click="$emit('openPrompt')">
          {{ promptName }}
        </button>
        <button v-else class="icon-btn" title="设置 Prompt" @click="$emit('openPrompt')">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- Messages Area -->
    <div class="messages-area" ref="messagesArea">
      <div v-if="displayMessages.length === 0 && !streaming" class="empty-state">
        <div class="empty-logo">⚡</div>
        <h2>Teaching Arena</h2>
        <p>配置模型并发送第一条消息</p>
      </div>

      <div class="message-feed">
        <div
          v-for="(msg, idx) in displayMessages"
          :key="idx"
          class="message-row"
          :class="`role-${msg.role}`"
        >
          <div v-if="msg.role === 'assistant'" class="avatar assistant-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a10 10 0 1 0 10 10H12z"/><path d="M12 2a10 10 0 0 0 0 20V2z"/>
            </svg>
          </div>
          <div class="message-bubble">
            <div class="markdown-body" v-html="renderMarkdown(msg.content)"></div>
          </div>
        </div>

        <!-- Streaming message -->
        <div v-if="streaming" class="message-row role-assistant">
          <div class="avatar assistant-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a10 10 0 1 0 10 10H12z"/><path d="M12 2a10 10 0 0 0 0 20V2z"/>
            </svg>
          </div>
          <div class="message-bubble">
            <div v-if="streamingText" class="markdown-body" v-html="renderMarkdown(streamingText)"></div>
            <div v-else class="typing-indicator">
              <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Error -->
      <div v-if="error" class="error-banner">
        <span>出现错误：{{ error }}</span>
      </div>
      <div class="bottom-spacer"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, nextTick, ref } from 'vue'
import { marked } from 'marked'

const props = defineProps<{
  side: string
  chatMode?: string
  modelName: string
  promptName: string
  messages: { role: string; content: string }[]
  responses: Record<number, string>
  streaming: boolean
  streamingText: string
  error: string
}>()

defineEmits<{ openPrompt: []; openModel: [] }>()

const messagesArea = ref<HTMLElement | null>(null)

marked.setOptions({ breaks: true, gfm: true })

function renderMarkdown(text: string): string {
  if (!text) return ''
  return marked.parse(text) as string
}

const displayMessages = computed(() => {
  const result: { role: string; content: string }[] = []
  for (let i = 0; i < props.messages.length; i++) {
    const msg = props.messages[i]
    if (msg.role === 'user') {
      result.push({ role: 'user', content: msg.content })
      if (props.responses[i] !== undefined) {
        result.push({ role: 'assistant', content: props.responses[i] })
      }
    }
  }
  return result
})

watch(
  [() => props.streamingText, () => displayMessages.value.length],
  () => {
    nextTick(() => {
      if (messagesArea.value) {
        messagesArea.value.scrollTop = messagesArea.value.scrollHeight
      }
    })
  }
)
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-primary);
  position: relative;
}

/* Header - Floating clean style */
.panel-header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--bg-primary); /* Invisible unless scrolling */
}

.panel-label {
  display: flex;
  align-items: center;
  gap: 12px;
}

.model-name {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.side-badge {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: var(--bg-input);
  padding: 2px 6px;
  border-radius: 4px;
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
  font-size: 0.8rem;
  padding: 4px 12px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.action-text-btn:hover {
  background: var(--bg-input);
  color: var(--text-primary);
}

.icon-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-tertiary);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.icon-btn:hover {
  background: var(--bg-input);
  color: var(--text-primary);
}

/* Messages Area */
.messages-area {
  flex: 1;
  overflow-y: auto;
  padding: 20px 0;
  display: flex;
  flex-direction: column;
}

.message-feed {
  display: flex;
  flex-direction: column;
  padding: 0 24px;
  max-width: 800px;
  margin: 0 auto;
  width: 100%;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-tertiary);
}
.empty-logo {
  font-size: 2.5rem;
  margin-bottom: 16px;
  opacity: 0.8;
}
.empty-state h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 8px;
}

/* ChatGPT Message Styles */
.message-row {
  display: flex;
  width: 100%;
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease;
}

.role-user {
  justify-content: flex-end;
}

.role-assistant {
  justify-content: flex-start;
}

/* User bubbles */
.role-user .message-bubble {
  background: var(--bg-user-bubble);
  color: var(--text-primary);
  padding: 12px 18px;
  border-radius: 20px;
  max-width: 85%;
  font-size: 1rem;
}

/* Assistant bubbles (virtually no bubble) */
.role-assistant .message-bubble {
  max-width: 100%;
  padding: 4px 0;
}

.avatar {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  border: 1px solid var(--border-primary);
}
.assistant-avatar {
  background: #ffffff;
  color: #000000;
  border: none;
}

/* Error */
.error-banner {
  margin: 16px auto;
  max-width: 800px;
  padding: 12px 16px;
  background: rgba(239, 68, 68, 0.1);
  border-radius: var(--radius-md);
  color: #ef4444;
  font-size: 0.9rem;
}

.bottom-spacer {
  height: 120px; /* Space for the floating input bar */
}
</style>
