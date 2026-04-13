<template>
  <div class="sidebar" :class="{ 'is-collapsed': collapsed }">
    <!-- Header -->
    <div class="sidebar-header">
      <button class="new-chat-btn" @click="$emit('new')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
        <span v-show="!collapsed">新建对话</span>
      </button>
      <button class="collapse-btn" @click="collapsed = !collapsed" :title="collapsed ? '展开侧边栏' : '收起侧边栏'">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <line x1="9" y1="3" x2="9" y2="21"/>
        </svg>
      </button>
    </div>

    <!-- Session List -->
    <div class="sidebar-content">
      <template v-if="!collapsed">
        <div v-if="sessions.length === 0" class="empty-list">暂无记录</div>
      <div
        v-for="session in sessions"
        :key="session.id"
        class="session-item"
        :class="{ 'is-active': session.id === currentId }"
        @click="$emit('switch', session.id)"
      >
        <svg class="message-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <div class="session-title">{{ session.title }}</div>
        
        <button class="delete-btn" @click.stop="$emit('delete', session.id)" title="删除">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      </div>
      </template>
    </div>

    <!-- Footer -->
    <div class="sidebar-footer">
      <button class="settings-btn" @click="$emit('openSettings')" :title="collapsed ? '设置' : ''">
        <svg class="settings-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        <span v-show="!collapsed" class="settings-text">API 与模型设置</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  sessions: { id: string; title: string; updatedAt: number }[]
  currentId: string | null
}>()

defineEmits<{
  new: []
  switch: [id: string]
  delete: [id: string]
  openSettings: []
}>()

const collapsed = ref(false)
</script>

<style scoped>
.sidebar {
  width: 260px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-primary);
  display: flex;
  flex-direction: column;
  transition: width var(--transition-base);
  flex-shrink: 0;
  position: relative;
}

.sidebar.is-collapsed {
  width: 60px;
}

.sidebar-header {
  padding: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.is-collapsed .sidebar-header {
  flex-direction: column;
  padding: 12px 0;
  align-items: center;
}

.new-chat-btn {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: transparent;
  border: 1px solid var(--border-primary);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all var(--transition-fast);
  overflow: hidden;
  white-space: nowrap;
}

.is-collapsed .new-chat-btn {
  padding: 0;
  width: 34px;
  height: 34px;
  min-width: 34px;
  justify-content: center;
  border: none;
}

.is-collapsed .new-chat-btn:hover {
  background: rgba(255,255,255,0.05);
}

.new-chat-btn:hover {
  background: rgba(255,255,255,0.05);
}

.collapse-btn {
  width: 34px;
  height: 34px;
  min-width: 34px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-sm);
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
}

.collapse-btn:hover {
  background: rgba(255,255,255,0.05);
  color: var(--text-primary);
}

.sidebar-content {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-bottom: 80px; /* Spacer for absolute footer */
}

/* Scrollbar adjustment for sidebar */
.sidebar-content::-webkit-scrollbar {
  width: 4px;
}

.empty-list {
  padding: 20px;
  text-align: center;
  color: var(--text-tertiary);
  font-size: 0.85rem;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  box-sizing: border-box;
  transition: all var(--transition-fast);
  group: hover;
}

.session-item:hover {
  background: rgba(255,255,255,0.04);
}

.session-item.is-active {
  background: rgba(255,255,255,0.08);
  color: var(--text-primary);
}

.message-icon {
  flex-shrink: 0;
  opacity: 0.7;
}

.session-title {
  flex: 1;
  font-size: 0.85rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.delete-btn {
  opacity: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: transparent;
  border: none;
  color: var(--text-tertiary);
  cursor: pointer;
  border-radius: 4px;
}

.session-item:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.sidebar-footer {
  padding: 12px;
  border-top: 1px solid var(--border-primary);
  display: flex;
  
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background: var(--bg-secondary);
  z-index: 10;
}

.settings-btn {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.settings-btn:hover {
  background: rgba(255,255,255,0.05);
  color: var(--text-primary);
}

.is-collapsed .settings-btn {
  justify-content: center;
  padding: 10px 0;
}

.settings-text {
  font-size: 0.85rem;
  font-family: var(--font-sans);
  white-space: nowrap;
}
</style>
