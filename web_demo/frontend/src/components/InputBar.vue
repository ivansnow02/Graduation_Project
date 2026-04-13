<template>
  <div class="input-floating-wrapper">
    <div class="input-container">
      <textarea
        ref="inputRef"
        v-model="inputText"
        placeholder="给 Teaching Arena 发送消息..."
        rows="1"
        @keydown.enter.exact="handleSend"
        @input="autoResize"
      ></textarea>
      
      <div class="input-actions">
        <button v-if="isStreaming" class="action-btn stop-btn" @click="$emit('stop')" title="停止生成">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>
          </svg>
        </button>
        <button v-else class="action-btn send-btn" :class="{ 'is-active': inputText.trim() }" :disabled="!inputText.trim()" @click="handleSend" title="发送">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="arrow-up" stroke-width="2" stroke="currentColor">
            <path d="M12 19V5M5 12l7-7 7 7"/>
          </svg>
        </button>
      </div>
    </div>
    <p class="input-hint">Teaching Arena 可以犯错。请核查重要信息。</p>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'

defineProps<{ isStreaming: boolean }>()
const emit = defineEmits<{ send: [content: string]; stop: [] }>()

const inputText = ref('')
const inputRef = ref<HTMLTextAreaElement | null>(null)

function handleSend(e?: KeyboardEvent) {
  if (e?.shiftKey) return
  e?.preventDefault()
  if (!inputText.value.trim() || emit.isStreaming) return
  emit('send', inputText.value.trim())
  inputText.value = ''
  nextTick(() => { if (inputRef.value) inputRef.value.style.height = 'auto' })
}

function autoResize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}
</script>

<style scoped>
.input-floating-wrapper {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0 24px 16px;
  background: linear-gradient(180deg, transparent 0%, var(--bg-primary) 50%);
  pointer-events: none; /* Let clicks pass through background */
  z-index: 50;
}

.input-container {
  pointer-events: auto; /* Re-enable clicks for the input itself */
  display: flex;
  align-items: flex-end;
  gap: 8px;
  width: 100%;
  max-width: 800px; /* Match ChatGPT width */
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: 8px 12px 8px 16px;
  box-shadow: var(--shadow-input);
  transition: border-color var(--transition-fast);
}

.input-container:focus-within {
  border-color: var(--border-secondary);
}

textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 1rem;
  line-height: 1.5;
  resize: none;
  min-height: 24px;
  max-height: 200px;
  padding: 6px 0;
  margin: 4px 0;
}

textarea::placeholder {
  color: var(--text-tertiary);
}

.input-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 2px;
}

.action-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all var(--transition-fast);
  background: #676767;
  color: var(--bg-primary);
}

.send-btn.is-active {
  background: #ECECEC;
  color: var(--bg-primary);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.send-btn:not(:disabled):hover {
  background: #FFFFFF;
}

.stop-btn {
  background: transparent;
  color: var(--text-secondary);
  border: 2px solid var(--text-secondary);
  width: 32px;
  height: 32px;
}

.stop-btn:hover {
  color: var(--text-primary);
  border-color: var(--text-primary);
}

.arrow-up {
  width: 18px;
  height: 18px;
}

.input-hint {
  pointer-events: auto;
  font-size: 0.75rem;
  color: var(--text-tertiary);
  margin-top: 10px;
  text-align: center;
}
</style>
