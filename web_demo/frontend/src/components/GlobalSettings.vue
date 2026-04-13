<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="visible" class="modal-overlay" @click.self="$emit('close')">
        <div class="modal glass-strong">
          <div class="modal-header">
            <h3>全局设置 (Global Settings)</h3>
            <button class="close-btn" @click="$emit('close')">&times;</button>
          </div>

          <div class="modal-body">
            <!-- Sidebar / Tabs for Settings (Simple 2 sections approach is easier) -->
            <div class="settings-layout">
              <div class="settings-sidebar">
                <button 
                  class="tab-btn" 
                  :class="{ active: activeTab === 'modelA' }"
                  @click="activeTab = 'modelA'"
                >
                  <span class="dot a"></span> Model A 模型
                </button>
                <button 
                  class="tab-btn" 
                  :class="{ active: activeTab === 'modelB' }"
                  @click="activeTab = 'modelB'"
                >
                  <span class="dot b"></span> Model B 模型
                </button>
              </div>

              <div class="settings-content">
                <div v-show="activeTab === 'modelA'" class="setting-pane">
                  <h4>Model A 参数设置</h4>
                  <p class="desc">控制左侧或单模型状态下的基准模型。</p>
                  
                  <div class="form-group">
                    <label>快捷选择</label>
                    <div class="quick-models">
                      <button
                        v-for="m in defaultModels"
                        :key="m.name"
                        class="quick-model-btn"
                        :class="{ active: localConfig.panelA.name === m.name }"
                        @click="selectQuickModel('panelA', m)"
                      >
                        {{ m.label }}
                      </button>
                    </div>
                  </div>

                  <div class="form-group">
                    <label>模型名称</label>
                    <input v-model="localConfig.panelA.name" type="text" placeholder="例如: qwen-plus" class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>API Key</label>
                    <input v-model="localConfig.panelA.apiKey" type="password" placeholder="sk-..." class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>Base URL</label>
                    <input v-model="localConfig.panelA.baseUrl" type="text" placeholder="默认使用 OpenAI 规范端点" class="input-field" />
                  </div>
                </div>

                <div v-show="activeTab === 'modelB'" class="setting-pane">
                  <h4>Model B 参数设置</h4>
                  <p class="desc">仅在 Side-by-Side 模式下作为对照模型出现。</p>
                  
                  <div class="form-group">
                    <label>快捷选择</label>
                    <div class="quick-models">
                      <button
                        v-for="m in defaultModels"
                        :key="m.name"
                        class="quick-model-btn"
                        :class="{ active: localConfig.panelB.name === m.name }"
                        @click="selectQuickModel('panelB', m)"
                      >
                        {{ m.label }}
                      </button>
                    </div>
                  </div>

                  <div class="form-group">
                    <label>模型名称</label>
                    <input v-model="localConfig.panelB.name" type="text" placeholder="例如: gpt-4o" class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>API Key</label>
                    <input v-model="localConfig.panelB.apiKey" type="password" placeholder="sk-..." class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>Base URL</label>
                    <input v-model="localConfig.panelB.baseUrl" type="text" placeholder="默认使用 OpenAI 规范端点" class="input-field" />
                  </div>
                </div>

              </div>
            </div>
          </div>

          <div class="modal-footer">
            <button class="btn-secondary" @click="$emit('close')">取消</button>
            <button class="btn-primary" @click="handleSave">保存设置</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, reactive, watch } from 'vue'
import { defaultModels, type QuickModel } from '../data/presets'
import type { GlobalModelConfig } from '../composables/useChat'

const props = defineProps<{
  visible: boolean
  config: GlobalModelConfig
}>()

const emit = defineEmits<{
  close: []
  save: [config: GlobalModelConfig]
}>()

const activeTab = ref<'modelA' | 'modelB'>('modelA')

const localConfig = reactive<GlobalModelConfig>({
  panelA: { name: '', apiKey: '', baseUrl: '' },
  panelB: { name: '', apiKey: '', baseUrl: '' }
})

watch(() => props.visible, (v) => {
  if (v) {
    localConfig.panelA = { ...props.config.panelA }
    localConfig.panelB = { ...props.config.panelB }
    activeTab.value = 'modelA'
  }
})

function selectQuickModel(panel: 'panelA' | 'panelB', m: QuickModel) {
  localConfig[panel].name = m.name
  localConfig[panel].baseUrl = m.baseUrl
}

function handleSave() {
  emit('save', {
    panelA: { ...localConfig.panelA },
    panelB: { ...localConfig.panelB }
  })
  emit('close')
}
</script>

<style scoped>
.modal-overlay {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,0.6); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
}
.modal {
  width: 720px; max-width: 95vw; height: 80vh; max-height: 700px;
  border-radius: var(--radius-xl); overflow: hidden;
  animation: slideUp 0.3s ease;
  display: flex; flex-direction: column;
}
.modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 24px; border-bottom: 1px solid var(--border-primary);
  flex-shrink: 0;
}
.modal-header h3 { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); }
.close-btn {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  background: transparent; border: none; color: var(--text-secondary);
  font-size: 1.2rem; cursor: pointer; border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
}
.close-btn:hover { background: rgba(255,255,255,0.06); color: var(--text-primary); }

.modal-body { 
  flex: 1; overflow: hidden; display: flex; 
}
.settings-layout {
  display: flex; width: 100%; height: 100%;
}
.settings-sidebar {
  width: 180px; background: var(--bg-secondary);
  border-right: 1px solid var(--border-primary);
  padding: 16px 8px;
  display: flex; flex-direction: column; gap: 4px;
}
.tab-btn {
  display: flex; align-items: center; gap: 10px;
  width: 100%; padding: 10px 14px;
  background: transparent; border: none;
  border-radius: var(--radius-md);
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.9rem; text-align: left;
  cursor: pointer; transition: all var(--transition-fast);
}
.tab-btn:hover { background: rgba(255,255,255,0.05); color: var(--text-primary); }
.tab-btn.active {
  background: rgba(255,255,255,0.08); color: var(--text-primary);
  font-weight: 500;
}
.dot { width: 8px; height: 8px; border-radius: 50%; opacity: 0.8; }
.dot.a { background: #6366f1; }
.dot.b { background: #10b981; }

.settings-content {
  flex: 1; padding: 24px; overflow-y: auto; background: var(--bg-primary);
}
.setting-pane h4 { font-size: 1.1rem; margin-bottom: 6px; }
.desc { color: var(--text-tertiary); font-size: 0.85rem; margin-bottom: 24px; }

.form-group { margin-bottom: 18px; }
.form-group label {
  display: block; font-size: 0.85rem; font-weight: 500;
  color: var(--text-secondary); margin-bottom: 8px;
}
.input-field {
  width: 80%; padding: 10px 14px; background: var(--bg-input);
  border: 1px solid var(--border-primary); border-radius: var(--radius-md);
  color: var(--text-primary); font-family: var(--font-sans); font-size: 0.875rem;
  outline: none; transition: all var(--transition-fast);
}
.input-field:focus { border-color: var(--accent-indigo); box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }

.quick-models { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.quick-model-btn {
  padding: 6px 14px; border-radius: var(--radius-full);
  border: 1px solid var(--border-secondary); background: transparent;
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.85rem; cursor: pointer; transition: all var(--transition-fast);
}
.quick-model-btn.active {
  border-color: var(--accent-indigo); background: rgba(99,102,241,0.15); color: #fff;
}

.modal-footer {
  display: flex; justify-content: flex-end; gap: 10px;
  padding: 16px 24px; border-top: 1px solid var(--border-primary);
  flex-shrink: 0; background: var(--bg-primary);
}
.btn-secondary {
  padding: 8px 20px; border-radius: var(--radius-md);
  border: 1px solid var(--border-secondary); background: transparent;
  color: var(--text-secondary); font-family: var(--font-sans);
  cursor: pointer;
}
.btn-primary {
  padding: 8px 20px; border-radius: var(--radius-md); border: none;
  background: var(--gradient-primary); color: #fff; font-weight: 500;
  cursor: pointer;
}
</style>
