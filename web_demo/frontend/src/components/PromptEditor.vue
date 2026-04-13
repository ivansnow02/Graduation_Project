<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
        <div class="modal glass-strong prompt-modal">
          <div class="modal-header">
            <h3>设置 System Prompt · 模型 {{ side === 'a' ? 'A' : 'B' }}</h3>
            <button class="close-btn" @click="$emit('close')">&times;</button>
          </div>

          <div class="modal-body">
            <div class="form-group">
              <label>预设策略</label>
              <div class="preset-list">
                <button
                  v-for="p in presetPrompts"
                  :key="p.id"
                  class="preset-item"
                  :class="{ active: selectedPresetId === p.id }"
                  @click="selectPreset(p)"
                >
                  <div class="preset-name">{{ p.name }}</div>
                  <div class="preset-desc">{{ p.description }}</div>
                </button>
              </div>
            </div>

            <div class="form-group">
              <label for="prompt-content">Prompt 内容 <span class="hint">（可自由编辑）</span></label>
              <textarea
                id="prompt-content"
                v-model="localContent"
                class="prompt-textarea"
                rows="8"
                placeholder="输入你的 System Prompt..."
              ></textarea>
            </div>
          </div>

          <div class="modal-footer">
            <button class="btn-secondary" @click="handleClear">清除</button>
            <button class="btn-primary" @click="handleSave">应用</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { presetPrompts, type PresetPrompt } from '../data/presets'

const props = defineProps<{
  visible: boolean
  side: string
  currentPrompt: string
}>()

const emit = defineEmits<{
  close: []
  save: [payload: { content: string; name: string }]
}>()

const selectedPresetId = ref('')
const localContent = ref('')

watch(() => props.visible, (v) => {
  if (v) {
    localContent.value = props.currentPrompt
    const match = presetPrompts.find(p => p.content === props.currentPrompt)
    selectedPresetId.value = match ? match.id : ''
  }
})

function selectPreset(p: PresetPrompt) {
  selectedPresetId.value = p.id
  if (p.id !== 'custom') {
    localContent.value = p.content
  }
}

function handleSave() {
  const name = presetPrompts.find(p => p.id === selectedPresetId.value)?.name || ''
  emit('save', { content: localContent.value, name })
  emit('close')
}

function handleClear() {
  localContent.value = ''
  selectedPresetId.value = ''
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.prompt-modal {
  width: 600px;
  max-width: 95vw;
  max-height: 85vh;
  border-radius: var(--radius-xl);
  overflow: hidden;
  animation: slideUp 0.3s ease;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-primary);
}

.modal-header h3 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.close-btn {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; border: none;
  color: var(--text-secondary); font-size: 1.2rem;
  cursor: pointer; border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
}
.close-btn:hover { background: rgba(255,255,255,0.06); color: var(--text-primary); }

.modal-body { padding: 20px 24px; overflow-y: auto; max-height: 60vh; }

.form-group { margin-bottom: 16px; }
.form-group label {
  display: block; font-size: 0.8rem; font-weight: 500;
  color: var(--text-secondary); margin-bottom: 8px;
}
.hint { font-weight: 400; color: var(--text-tertiary); }

.preset-list {
  display: flex; flex-direction: column; gap: 6px;
  max-height: 240px; overflow-y: auto;
}

.preset-item {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
  padding: 10px 14px; background: rgba(0,0,0,0.2);
  border: 1px solid var(--border-primary); border-radius: var(--radius-md);
  cursor: pointer; transition: all var(--transition-fast);
  text-align: left; font-family: var(--font-sans);
}
.preset-item:hover { border-color: var(--border-secondary); background: rgba(0,0,0,0.3); }
.preset-item.active { border-color: var(--accent-indigo); background: rgba(99,102,241,0.08); }

.preset-name { font-size: 0.85rem; font-weight: 500; color: var(--text-primary); }
.preset-desc { font-size: 0.75rem; color: var(--text-tertiary); }

.prompt-textarea {
  width: 100%; padding: 12px 14px;
  background: var(--bg-input); border: 1px solid var(--border-primary);
  border-radius: var(--radius-md); color: var(--text-primary);
  font-family: var(--font-sans); font-size: 0.85rem; line-height: 1.6;
  resize: vertical; outline: none; min-height: 120px;
  transition: all var(--transition-fast);
}
.prompt-textarea:focus { border-color: var(--accent-indigo); box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
.prompt-textarea::placeholder { color: var(--text-tertiary); }

.modal-footer {
  display: flex; justify-content: flex-end; gap: 10px;
  padding: 16px 24px; border-top: 1px solid var(--border-primary);
}
.btn-secondary {
  padding: 8px 20px; border-radius: var(--radius-md);
  border: 1px solid var(--border-secondary); background: transparent;
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.85rem; cursor: pointer; transition: all var(--transition-fast);
}
.btn-secondary:hover { border-color: var(--text-tertiary); color: var(--text-primary); }
.btn-primary {
  padding: 8px 20px; border-radius: var(--radius-md); border: none;
  background: var(--gradient-primary); color: #fff;
  font-family: var(--font-sans); font-size: 0.85rem; font-weight: 500;
  cursor: pointer; transition: all var(--transition-fast);
  box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99,102,241,0.4); }
</style>
