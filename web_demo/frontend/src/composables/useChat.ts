import { ref, reactive, watch, onMounted, computed } from 'vue'
import { useLLM, type ModelConfig, type ChatMessage } from './useLLM'

export interface PanelState {
  systemPrompt: string
  promptName: string
  responses: Record<number, string>
  streaming: boolean
  streamingText: string
  error: string
  abortController: AbortController | null
}

export type ChatMode = 'single' | 'sbs'

export interface ChatSession {
  id: string
  title: string
  updatedAt: number
  mode: ChatMode
  messages: { role: 'user'; content: string }[]
  panelA: { systemPrompt: string; promptName: string; responses: Record<number, string> }
  panelB: { systemPrompt: string; promptName: string; responses: Record<number, string> }
}

export interface GlobalModelConfig {
  panelA: ModelConfig
  panelB: ModelConfig
}

const STORAGE_KEY_SESSIONS = 'teaching_arena_sessions_v3' // Updated key to avoid conflicts
const STORAGE_KEY_CONFIG = 'teaching_arena_config_v2'

function generateId() {
  return Math.random().toString(36).substring(2, 9)
}

function createEmptyPanelState(): PanelState {
  return reactive({
    systemPrompt: '',
    promptName: '',
    responses: {},
    streaming: false,
    streamingText: '',
    error: '',
    abortController: null,
  })
}

// Global Configs
const globalModelConfig = reactive<GlobalModelConfig>({
  panelA: { name: '', apiKey: '', baseUrl: '' },
  panelB: { name: '', apiKey: '', baseUrl: '' }
})

const sessions = ref<ChatSession[]>([])
const currentSessionId = ref<string | null>(null)
const chatMode = ref<ChatMode>('sbs')

export function useChat() {
  const { streamChat } = useLLM()

  // Current Active State
  const messages = ref<{ role: 'user'; content: string }[]>([])
  const panelA = createEmptyPanelState()
  const panelB = createEmptyPanelState()

  // Computed list of sessions for current mode
  const activeModeSessions = computed(() => {
    return sessions.value.filter(s => s.mode === chatMode.value)
  })

  // --- Persistence Logic ---

  function saveConfigToStorage() {
    try {
      localStorage.setItem(STORAGE_KEY_CONFIG, JSON.stringify(globalModelConfig))
    } catch (e) {
      console.warn('保存全局配置失败', e)
    }
  }

  function saveSessionsToStorage() {
    try {
      // First, update the current session in the list
      if (currentSessionId.value) {
        const idx = sessions.value.findIndex(s => s.id === currentSessionId.value)
        if (idx !== -1) {
          const s = sessions.value[idx]
          s.messages = [...messages.value]
          s.panelA.responses = { ...panelA.responses }
          s.panelA.systemPrompt = panelA.systemPrompt
          s.panelA.promptName = panelA.promptName
          s.panelB.responses = { ...panelB.responses }
          s.panelB.systemPrompt = panelB.systemPrompt
          s.panelB.promptName = panelB.promptName
          
          if (s.title === '新会话' && messages.value.length > 0) {
            s.title = messages.value[0].content.substring(0, 20) + (messages.value[0].content.length > 20 ? '...' : '')
          }
          s.updatedAt = Date.now()
        }
      }
      localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions.value))
    } catch (e) {
      console.warn('保存会话记录失败', e)
    }
  }

  function loadFromStorage() {
    try {
      const configStr = localStorage.getItem(STORAGE_KEY_CONFIG)
      if (configStr) {
        const config = JSON.parse(configStr)
        if (config.panelA) Object.assign(globalModelConfig.panelA, config.panelA)
        if (config.panelB) Object.assign(globalModelConfig.panelB, config.panelB)
      }

      const sessionsStr = localStorage.getItem(STORAGE_KEY_SESSIONS)
      if (sessionsStr) {
        sessions.value = JSON.parse(sessionsStr)
      }

      // Try switching to a session that matches the initial state
      if (activeModeSessions.value.length === 0) {
        createNewSession()
      } else {
        const latest = [...activeModeSessions.value].sort((a, b) => b.updatedAt - a.updatedAt)[0]
        switchSession(latest.id)
      }
    } catch (e) {
      console.warn('加载数据失败', e)
      createNewSession()
    }
  }

  // --- Session Management ---

  function setChatMode(mode: ChatMode) {
    if (chatMode.value === mode) return
    chatMode.value = mode
    
    // Switch to to the latest session in this mode, or create new
    if (activeModeSessions.value.length > 0) {
      const latest = [...activeModeSessions.value].sort((a, b) => b.updatedAt - a.updatedAt)[0]
      switchSession(latest.id)
    } else {
      createNewSession()
    }
  }

  function createNewSession() {
    stopStreaming()
    const newSession: ChatSession = {
      id: generateId(),
      title: '新会话',
      updatedAt: Date.now(),
      mode: chatMode.value,
      messages: [],
      panelA: { systemPrompt: '', promptName: '', responses: {} },
      panelB: { systemPrompt: '', promptName: '', responses: {} }
    }
    sessions.value.unshift(newSession)
    switchSession(newSession.id)
  }

  function switchSession(id: string) {
    stopStreaming()
    const session = sessions.value.find(s => s.id === id)
    if (!session) return

    currentSessionId.value = id
    messages.value = [...session.messages]
    
    panelA.responses = { ...session.panelA.responses }
    panelA.systemPrompt = session.panelA.systemPrompt
    panelA.promptName = session.panelA.promptName
    panelA.error = ''
    panelA.streamingText = ''
    
    panelB.responses = { ...session.panelB.responses }
    panelB.systemPrompt = session.panelB.systemPrompt
    panelB.promptName = session.panelB.promptName
    panelB.error = ''
    panelB.streamingText = ''
  }

  function deleteSession(id: string) {
    if (activeModeSessions.value.length === 1 && activeModeSessions.value[0].messages.length === 0) return
    
    sessions.value = sessions.value.filter(s => s.id !== id)
    
    if (activeModeSessions.value.length === 0) {
        createNewSession()
    } else if (currentSessionId.value === id) {
        switchSession(activeModeSessions.value[0].id)
    }
    saveSessionsToStorage()
  }

  // Auto load
  onMounted(() => {
    // Only load once
    if (sessions.value.length === 0 && !currentSessionId.value) {
      loadFromStorage()
    }
  })

  // Watchers for autosave
  watch(globalModelConfig, saveConfigToStorage, { deep: true })
  watch(
    [
      messages,
      () => panelA.responses, () => panelA.systemPrompt, () => panelA.promptName,
      () => panelB.responses, () => panelB.systemPrompt, () => panelB.promptName
    ],
    () => saveSessionsToStorage(),
    { deep: true }
  )


  // --- Context Compression & Chat Logic ---

  function buildHistory(panel: PanelState, maxRounds = 10): ChatMessage[] {
    const history: ChatMessage[] = []
    
    if (panel.systemPrompt) {
      history.push({ role: 'system', content: panel.systemPrompt })
    }

    const totalMessages = messages.value.length
    const startIndex = Math.max(0, totalMessages - maxRounds)

    for (let i = startIndex; i < totalMessages; i++) {
      history.push({ role: 'user', content: messages.value[i].content })
      if (panel.responses[i] !== undefined) {
        history.push({ role: 'assistant', content: panel.responses[i] })
      }
    }
    return history
  }

  function streamToPanel(panelId: 'a' | 'b', msgIndex: number, temperature: number, maxTokens: number) {
    const panel = panelId === 'a' ? panelA : panelB
    const modelCfg = panelId === 'a' ? globalModelConfig.panelA : globalModelConfig.panelB

    if (!modelCfg.name || !modelCfg.apiKey || !modelCfg.baseUrl) {
      panel.error = '请配置完模型后再试'
      return
    }

    panel.streaming = true
    panel.streamingText = ''
    panel.error = ''

    const controller = new AbortController()
    panel.abortController = controller

    const history = buildHistory(panel, 10)

    streamChat(
      modelCfg,
      history,
      {
        onChunk(chunk) {
          panel.streamingText += chunk
        },
        onDone() {
          panel.responses[msgIndex] = panel.streamingText
          panel.streaming = false
          panel.streamingText = ''
          panel.abortController = null
        },
        onError(err) {
          panel.error = err
          if (panel.streamingText) {
            panel.responses[msgIndex] = panel.streamingText
          }
          panel.streaming = false
          panel.streamingText = ''
          panel.abortController = null
        },
      },
      { temperature, maxTokens },
      controller.signal
    )
  }

  function sendMessage(content: string, targetPanels: ('a' | 'b')[] = ['a', 'b'], temperature = 0.8, maxTokens = 2048) {
    if (!content.trim()) return
    if (panelA.streaming || panelB.streaming) return

    const msgIndex = messages.value.length
    messages.value.push({ role: 'user', content: content.trim() })

    if (targetPanels.includes('a')) streamToPanel('a', msgIndex, temperature, maxTokens)
    if (targetPanels.includes('b')) streamToPanel('b', msgIndex, temperature, maxTokens)
  }

  function stopStreaming() {
    for (const panel of [panelA, panelB]) {
      if (panel.abortController) {
        panel.abortController.abort()
        if (panel.streamingText) {
          panel.responses[messages.value.length - 1] = panel.streamingText
        }
        panel.streaming = false
        panel.streamingText = ''
        panel.abortController = null
      }
    }
  }

  function clearChat() {
    stopStreaming()
    messages.value = []
    for (const panel of [panelA, panelB]) {
      panel.responses = {}
      panel.streamingText = ''
      panel.error = ''
    }
  }

  return {
    messages, panelA, panelB, sendMessage, stopStreaming, clearChat,
    globalModelConfig,
    chatMode, setChatMode,
    sessions, activeModeSessions, currentSessionId, createNewSession, switchSession, deleteSession
  }
}
