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
                  :class="{ active: activeTab === 'profiles' }"
                  @click="activeTab = 'profiles'"
                >
                  <span class="dot profiles"></span> 模型配置库
                </button>
                <button 
                  class="tab-btn" 
                  :class="{ active: activeTab === 'modelA' }"
                  @click="activeTab = 'modelA'"
                >
                  <span class="dot a"></span> 基准模型
                </button>
                <button 
                  class="tab-btn" 
                  :class="{ active: activeTab === 'modelB' }"
                  @click="activeTab = 'modelB'"
                >
                  <span class="dot b"></span> 对照模型
                </button>
                <button
                  class="tab-btn"
                  :class="{ active: activeTab === 'evaluator' }"
                  @click="activeTab = 'evaluator'"
                >
                  <span class="dot eval"></span> 评测模型
                </button>
                <button
                  class="tab-btn"
                  :class="{ active: activeTab === 'generation' }"
                  @click="activeTab = 'generation'"
                >
                  <span class="dot generation"></span> 生成参数
                </button>
              </div>

              <div class="settings-content">
                <div v-show="activeTab === 'profiles'" class="setting-pane">
                  <h4>模型配置库</h4>
                  <p class="desc">统一保存模型名称、API Key 与 Base URL，基准、对照和评测模型均从此处选择。</p>

                  <div class="profile-list">
                    <button
                      v-for="profile in localConfig.modelProfiles"
                      :key="profile.id"
                      class="profile-item"
                      :class="{ active: editingProfileId === profile.id }"
                      @click="editProfile(profile)"
                    >
                      <span class="profile-label">{{ profile.label }}</span>
                      <span class="profile-meta">{{ profile.name || '未填写模型名称' }}</span>
                    </button>
                  </div>

                  <div class="profile-editor">
                    <div class="form-group">
                      <label>配置名称</label>
                      <input v-model="profileDraft.label" type="text" placeholder="例如：Qwen Plus 主模型" class="input-field" />
                    </div>
                    <div class="form-group">
                      <label>模型名称</label>
                      <input v-model="profileDraft.name" type="text" placeholder="例如: qwen-plus" class="input-field" />
                    </div>
                    <div class="form-group">
                      <label>API Key</label>
                      <input v-model="profileDraft.apiKey" type="password" placeholder="sk-..." class="input-field" />
                    </div>
                    <div class="form-group">
                      <label>Base URL</label>
                      <input v-model="profileDraft.baseUrl" type="text" placeholder="例如: https://dashscope.aliyuncs.com/compatible-mode/v1" class="input-field" />
                    </div>

                    <div class="profile-actions">
                      <button class="btn-secondary" @click="resetProfileDraft">新建配置</button>
                      <button class="btn-secondary danger" :disabled="!editingProfileId" @click="deleteProfile">删除配置</button>
                      <button class="btn-primary" @click="saveProfile">保存配置</button>
                    </div>
                  </div>
                </div>

                <div v-show="activeTab === 'modelA'" class="setting-pane">
                  <h4>基准模型参数设置</h4>
                  <p class="desc">用于单模型评测，并在双模型对照中作为左侧基准。</p>
                  
                  <div class="form-group">
                    <label>选择模型配置</label>
                    <select v-model="localConfig.panelAProfileId" class="input-field" @change="selectProfile('panelA', localConfig.panelAProfileId)">
                      <option value="">未选择</option>
                      <option v-for="profile in localConfig.modelProfiles" :key="profile.id" :value="profile.id">
                        {{ profile.label }}
                      </option>
                    </select>
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
                  <h4>对照模型参数设置</h4>
                  <p class="desc">仅在双模型对照模式下作为右侧比较对象。</p>
                  
                  <div class="form-group">
                    <label>选择模型配置</label>
                    <select v-model="localConfig.panelBProfileId" class="input-field" @change="selectProfile('panelB', localConfig.panelBProfileId)">
                      <option value="">未选择</option>
                      <option v-for="profile in localConfig.modelProfiles" :key="profile.id" :value="profile.id">
                        {{ profile.label }}
                      </option>
                    </select>
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

                <div v-show="activeTab === 'evaluator'" class="setting-pane">
                  <h4>评测模型参数设置</h4>
                  <p class="desc">用于给当前聊天轨迹生成 annotations，再计算客观评测指标。</p>

                  <div class="form-group">
                    <label>选择模型配置</label>
                    <select v-model="localConfig.evaluatorProfileId" class="input-field" @change="selectProfile('evaluator', localConfig.evaluatorProfileId)">
                      <option value="">未选择</option>
                      <option v-for="profile in localConfig.modelProfiles" :key="profile.id" :value="profile.id">
                        {{ profile.label }}
                      </option>
                    </select>
                  </div>

                  <div class="form-group">
                    <label>模型名称</label>
                    <input v-model="localConfig.evaluator.name" type="text" placeholder="例如: qwen-plus" class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>API Key</label>
                    <input v-model="localConfig.evaluator.apiKey" type="password" placeholder="sk-..." class="input-field" />
                  </div>
                  <div class="form-group">
                    <label>Base URL</label>
                    <input v-model="localConfig.evaluator.baseUrl" type="text" placeholder="默认使用 OpenAI 规范端点" class="input-field" />
                  </div>
                </div>

                <div v-show="activeTab === 'generation'" class="setting-pane">
                  <h4>聊天生成参数</h4>
                  <p class="desc">控制模型回复的随机性。温度越低，输出越稳定；温度越高，表达越发散。</p>

                  <div class="form-group">
                    <div class="range-label">
                      <label for="temperature-range">聊天温度</label>
                      <span class="range-value">{{ localConfig.generation.temperature.toFixed(2) }}</span>
                    </div>
                    <input
                      id="temperature-range"
                      v-model.number="localConfig.generation.temperature"
                      class="range-field"
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                    />
                    <input
                      v-model.number="localConfig.generation.temperature"
                      class="input-field compact-field"
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                    />
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
import type { GlobalModelConfig, ModelProfile } from '../composables/useChat'

const props = defineProps<{
  visible: boolean
  config: GlobalModelConfig
}>()

const emit = defineEmits<{
  close: []
  save: [config: GlobalModelConfig]
}>()

const activeTab = ref<'profiles' | 'modelA' | 'modelB' | 'evaluator' | 'generation'>('profiles')

const localConfig = reactive<GlobalModelConfig>({
  panelA: { name: '', apiKey: '', baseUrl: '' },
  panelB: { name: '', apiKey: '', baseUrl: '' },
  evaluator: { name: '', apiKey: '', baseUrl: '' },
  modelProfiles: [],
  panelAProfileId: '',
  panelBProfileId: '',
  evaluatorProfileId: '',
  generation: { temperature: 0.8 }
})

const editingProfileId = ref('')
const profileDraft = reactive<Omit<ModelProfile, 'id'>>({
  label: '',
  name: '',
  apiKey: '',
  baseUrl: '',
})

watch(() => props.visible, (v) => {
  if (v) {
    localConfig.panelA = { ...props.config.panelA }
    localConfig.panelB = { ...props.config.panelB }
    localConfig.evaluator = { ...props.config.evaluator }
    localConfig.modelProfiles = props.config.modelProfiles.map(profile => ({ ...profile }))
    localConfig.panelAProfileId = props.config.panelAProfileId
    localConfig.panelBProfileId = props.config.panelBProfileId
    localConfig.evaluatorProfileId = props.config.evaluatorProfileId
    localConfig.generation = { ...props.config.generation }
    activeTab.value = 'profiles'
    resetProfileDraft()
  }
})

function generateProfileId() {
  return Math.random().toString(36).slice(2, 9)
}

function resetProfileDraft() {
  editingProfileId.value = ''
  profileDraft.label = ''
  profileDraft.name = ''
  profileDraft.apiKey = ''
  profileDraft.baseUrl = ''
}

function editProfile(profile: ModelProfile) {
  editingProfileId.value = profile.id
  profileDraft.label = profile.label
  profileDraft.name = profile.name
  profileDraft.apiKey = profile.apiKey
  profileDraft.baseUrl = profile.baseUrl
}

function saveProfile() {
  const label = profileDraft.label.trim() || profileDraft.name.trim() || '未命名模型'
  const payload = {
    label,
    name: profileDraft.name.trim(),
    apiKey: profileDraft.apiKey.trim(),
    baseUrl: profileDraft.baseUrl.trim(),
  }

  if (editingProfileId.value) {
    const idx = localConfig.modelProfiles.findIndex(profile => profile.id === editingProfileId.value)
    if (idx !== -1) {
      localConfig.modelProfiles[idx] = { id: editingProfileId.value, ...payload }
      syncSelectedProfile(editingProfileId.value)
    }
    return
  }

  const id = generateProfileId()
  localConfig.modelProfiles.push({ id, ...payload })
  editingProfileId.value = id
}

function deleteProfile() {
  if (!editingProfileId.value) return
  const id = editingProfileId.value
  localConfig.modelProfiles = localConfig.modelProfiles.filter(profile => profile.id !== id)
  for (const target of ['panelA', 'panelB', 'evaluator'] as const) {
    if (localConfig[`${target}ProfileId`] === id) {
      localConfig[`${target}ProfileId`] = ''
    }
  }
  resetProfileDraft()
}

function selectProfile(target: 'panelA' | 'panelB' | 'evaluator', profileId: string) {
  localConfig[`${target}ProfileId`] = profileId
  const profile = localConfig.modelProfiles.find(item => item.id === profileId)
  if (!profile) return
  localConfig[target] = {
    name: profile.name,
    apiKey: profile.apiKey,
    baseUrl: profile.baseUrl,
  }
}

function syncSelectedProfile(profileId: string) {
  for (const target of ['panelA', 'panelB', 'evaluator'] as const) {
    if (localConfig[`${target}ProfileId`] === profileId) {
      selectProfile(target, profileId)
    }
  }
}

function handleSave() {
  const temperature = Math.min(1, Math.max(0, Number(localConfig.generation.temperature) || 0))
  emit('save', {
    panelA: { ...localConfig.panelA },
    panelB: { ...localConfig.panelB },
    evaluator: { ...localConfig.evaluator },
    modelProfiles: localConfig.modelProfiles.map(profile => ({ ...profile })),
    panelAProfileId: localConfig.panelAProfileId,
    panelBProfileId: localConfig.panelBProfileId,
    evaluatorProfileId: localConfig.evaluatorProfileId,
    generation: { temperature }
  })
  emit('close')
}
</script>

<style scoped>
.modal-overlay {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0, 0, 0, 0.68); backdrop-filter: blur(10px);
  display: flex; align-items: center; justify-content: center;
}
.modal {
  width: 720px; max-width: 95vw; height: 80vh; max-height: 700px;
  border-radius: var(--radius-lg); overflow: hidden;
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
  border-radius: var(--radius-sm);
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.9rem; text-align: left;
  cursor: pointer; transition: all var(--transition-fast);
}
.tab-btn:hover { background: rgba(255,255,255,0.06); color: var(--text-primary); }
.tab-btn.active {
  background: rgba(255,255,255,0.11); color: var(--text-primary);
  font-weight: 500;
}
.dot { width: 8px; height: 8px; border-radius: 50%; opacity: 0.8; }
.dot.a { background: var(--accent-cyan); }
.dot.b { background: var(--accent-success); }
.dot.eval { background: var(--accent-warning); }
.dot.generation { background: var(--text-secondary); }
.dot.profiles { background: var(--border-accent); }

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
  width: min(100%, 520px); padding: 11px 14px; background: var(--bg-input);
  border: 1px solid var(--border-primary); border-radius: var(--radius-md);
  color: var(--text-primary); font-family: var(--font-sans); font-size: 0.875rem;
  outline: none; transition: all var(--transition-fast);
}
.input-field:focus { border-color: var(--border-accent); box-shadow: 0 0 0 3px rgba(255,255,255,0.08); }

.compact-field {
  margin-top: 10px;
  max-width: 120px;
}

.range-label {
  align-items: center;
  display: flex;
  justify-content: space-between;
  max-width: 520px;
}

.range-value {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 0.84rem;
}

.range-field {
  accent-color: #f1f1f1;
  display: block;
  margin-top: 6px;
  max-width: 520px;
  width: 100%;
}

.profile-list {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 20px;
}

.profile-item {
  align-items: flex-start;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  font-family: var(--font-sans);
  gap: 4px;
  min-height: 68px;
  padding: 10px 12px;
  text-align: left;
  transition: all var(--transition-fast);
}

.profile-item:hover,
.profile-item.active {
  background: rgba(255, 255, 255, 0.08);
  border-color: var(--border-secondary);
  color: var(--text-primary);
}

.profile-label {
  font-size: 0.88rem;
  font-weight: 600;
}

.profile-meta {
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 100%;
}

.profile-editor {
  border-top: 1px solid var(--border-primary);
  padding-top: 18px;
}

.profile-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.quick-models { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.quick-model-btn {
  padding: 6px 14px; border-radius: var(--radius-full);
  border: 1px solid var(--border-secondary); background: transparent;
  color: var(--text-secondary); font-family: var(--font-sans);
  font-size: 0.85rem; cursor: pointer; transition: all var(--transition-fast);
}
.quick-model-btn.active {
  border-color: var(--border-accent); background: rgba(255,255,255,0.12); color: #fff;
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

.btn-secondary:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.btn-secondary.danger:not(:disabled):hover {
  border-color: rgba(243, 140, 140, 0.45);
  color: var(--accent-danger);
}
.btn-primary {
  padding: 8px 20px; border-radius: var(--radius-md); border: none;
  background: var(--gradient-primary); color: #171717; font-weight: 650;
  cursor: pointer;
}

@media (max-width: 720px) {
  .settings-layout {
    flex-direction: column;
  }

  .settings-sidebar {
    width: 100%;
    flex-direction: row;
    overflow-x: auto;
  }

  .tab-btn {
    white-space: nowrap;
  }
}
</style>
