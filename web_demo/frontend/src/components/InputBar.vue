<template>
  <div class="input-floating-wrapper">
    <div class="input-container">
      <textarea
        ref="inputRef"
        v-model="inputText"
        class="composer-input"
        placeholder="输入学生问题或教学情境..."
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
        <button v-else class="action-btn send-btn" :class="{ 'is-active': inputText.trim() }" :disabled="!inputText.trim()" @click="handleSend()" title="发送">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="arrow-up" stroke-width="2" stroke="currentColor">
            <path d="M12 19V5M5 12l7-7 7 7"/>
          </svg>
        </button>
      </div>
    </div>
    <p class="input-hint">模型输出仅供教学研究与实验分析参考，重要结论仍需人工核验。</p>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted, watch } from 'vue'

const props = defineProps<{ isStreaming: boolean }>()
const emit = defineEmits<{ send: [content: string]; stop: [] }>()

const inputText = ref('')
const inputRef = ref<HTMLTextAreaElement | null>(null)

function handleSend(e?: KeyboardEvent) {
  if (e?.shiftKey) return
  e?.preventDefault()
  if (!inputText.value.trim() || props.isStreaming) return
  emit('send', inputText.value.trim())
  inputText.value = ''
  nextTick(() => {
    if (!inputRef.value) return
    inputRef.value.style.height = 'auto'
    inputRef.value.focus()
  })
}

function autoResize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}

onMounted(() => {
  inputRef.value?.focus()
})

watch(() => props.isStreaming, (isStreaming) => {
  if (!isStreaming) {
    nextTick(() => inputRef.value?.focus())
  }
})
</script>

<style scoped>
.input-floating-wrapper {
  position: absolute;
  --composer-safe-area: 220px;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 34px 24px 16px;
  background: linear-gradient(180deg, transparent 0%, rgba(31, 31, 31, 0.9) 38%, var(--bg-primary) 72%);
  pointer-events: none;
  z-index: 50;
}

.input-container {
  pointer-events: auto;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  width: 100%;
  max-width: 820px;
  background: var(--bg-input);
  border: 1px solid var(--border-primary);
  border-radius: 26px;
  padding: 9px 11px 9px 18px;
  box-shadow: var(--shadow-input);
  transition: background var(--transition-fast), border-color var(--transition-fast), box-shadow var(--transition-fast);
}

.input-container:focus-within {
  border-color: var(--border-secondary);
  background: var(--bg-input-hover);
  box-shadow: 0 20px 54px rgba(0, 0, 0, 0.38);
}

.composer-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 0.98rem;
  line-height: 1.5;
  resize: none;
  min-height: 24px;
  max-height: 200px;
  padding: 6px 0;
  margin: 4px 0;
}

.composer-input::placeholder {
  color: var(--text-tertiary);
}

.input-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 2px;
}

.action-btn {
  width: 34px;
  height: 34px;
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
  background: #f1f1f1;
  color: var(--bg-primary);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.send-btn:not(:disabled):hover {
  background: #ffffff;
  transform: translateY(-1px);
}

.stop-btn {
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-secondary);
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
  width: 100%;
}

@media (max-width: 768px) {
  .input-floating-wrapper {
    --composer-safe-area: 200px;
    padding: 28px 12px 12px;
  }

  .input-container {
    border-radius: 22px;
    padding-left: 14px;
  }

  .input-hint {
    font-size: 0.7rem;
  }
}
</style>
