import { ref, reactive, watch, onMounted, computed } from "vue";
import { useLLM, type ModelConfig, type ChatMessage } from "./useLLM";
import {
  buildEvaluationDialogue,
  calculateObjectiveMetrics,
  cloneAnnotations,
  createEvaluationSourceHash,
  useEvaluation,
  type AnnotationFieldKey,
  type ObjectiveAnnotation,
  type EvaluationRecord,
} from "./useEvaluation";
import { presetPrompts } from "../data/presets";

export interface PanelState {
  systemPrompt: string;
  promptName: string;
  responses: Record<number, string>;
  streaming: boolean;
  streamingText: string;
  error: string;
  abortController: AbortController | null;
}

export type ChatMode = "single" | "sbs";

export interface ChatSession {
  id: string;
  title: string;
  updatedAt: number;
  mode: ChatMode;
  messages: { role: "user"; content: string }[];
  panelA: {
    systemPrompt: string;
    promptName: string;
    responses: Record<number, string>;
  };
  panelB: {
    systemPrompt: string;
    promptName: string;
    responses: Record<number, string>;
  };
  evaluation?: EvaluationRecord;
}

export interface AnnotationSelection {
  index: number;
  field: AnnotationFieldKey | null;
}

export interface GlobalModelConfig {
  panelA: ModelConfig;
  panelB: ModelConfig;
  evaluator: ModelConfig;
  modelProfiles: ModelProfile[];
  panelAProfileId: string;
  panelBProfileId: string;
  evaluatorProfileId: string;
  generation: {
    temperature: number;
  };
}

export interface ModelProfile extends ModelConfig {
  id: string;
  label: string;
}

const STORAGE_KEY_SESSIONS = "teaching_arena_sessions_v3"; // 更新的存储键，避免冲突
const STORAGE_KEY_CONFIG = "teaching_arena_config_v2";
const DEFAULT_PROMPT = presetPrompts.find((p) => p.id === "socratic_base")!;

function generateId() {
  return Math.random().toString(36).substring(2, 9);
}

function hasModelConfig(model: ModelConfig) {
  return Boolean(model.name && model.apiKey && model.baseUrl);
}

function buildTimestampToken(timestamp: number) {
  const date = new Date(timestamp);
  const parts = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0"),
  ];
  return `${parts[0]}${parts[1]}${parts[2]}-${parts[3]}${parts[4]}${parts[5]}`;
}

function downloadTextFile(content: string, filename: string) {
  const blob = new Blob([content], { type: "application/jsonl;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function containsCompletionMarker(text: string) {
  return /\[结束\]/.test(text);
}

function createEmptyPanelState(): PanelState {
  return reactive({
    systemPrompt: DEFAULT_PROMPT.content,
    promptName: DEFAULT_PROMPT.name,
    responses: {},
    streaming: false,
    streamingText: "",
    error: "",
    abortController: null,
  });
}

// 全局配置
const globalModelConfig = reactive<GlobalModelConfig>({
  panelA: { name: "", apiKey: "", baseUrl: "" },
  panelB: { name: "", apiKey: "", baseUrl: "" },
  evaluator: { name: "", apiKey: "", baseUrl: "" },
  modelProfiles: [],
  panelAProfileId: "",
  panelBProfileId: "",
  evaluatorProfileId: "",
  generation: { temperature: 0.8 },
});

const sessions = ref<ChatSession[]>([]);
const currentSessionId = ref<string | null>(null);
const chatMode = ref<ChatMode>("sbs");

export function useChat() {
  const { streamChat } = useLLM();
  const { evaluateDialogue } = useEvaluation();

  // 当前活动状态
  const messages = ref<{ role: "user"; content: string }[]>([]);
  const panelA = createEmptyPanelState();
  const panelB = createEmptyPanelState();
  const evaluation = reactive<{
    running: boolean;
    error: string;
    result: EvaluationRecord | null;
  }>({
    running: false,
    error: "",
    result: null,
  });
  const selectedAnnotation = reactive<AnnotationSelection>({
    index: -1,
    field: null,
  });

  // 当前模式下会话的计算列表
  const activeModeSessions = computed(() => {
    return sessions.value.filter((s) => s.mode === chatMode.value);
  });

  const evaluationDialogue = computed(() =>
    buildEvaluationDialogue(messages.value, panelA.responses),
  );
  const evaluationSourceHash = computed(() => {
    if (evaluationDialogue.value.length === 0) return "";
    return createEvaluationSourceHash(messages.value, panelA.responses);
  });
  const evaluationStale = computed(() => {
    return Boolean(
      evaluation.result &&
      evaluation.result.sourceHash !== evaluationSourceHash.value,
    );
  });
  const canRunEvaluation = computed(() => {
    return (
      chatMode.value === "single" &&
      !evaluation.running &&
      !panelA.streaming &&
      !panelB.streaming &&
      evaluationDialogue.value.length >= 2
    );
  });
  const canExportEvaluation = computed(() => {
    return (
      chatMode.value === "single" &&
      !evaluation.running &&
      Boolean(evaluation.result) &&
      !evaluationStale.value
    );
  });
  const panelACompleted = computed(() =>
    Object.values(panelA.responses).some(containsCompletionMarker),
  );
  const panelBCompleted = computed(() =>
    Object.values(panelB.responses).some(containsCompletionMarker),
  );
  const isDialogueCompleted = computed(() => {
    return chatMode.value === "sbs"
      ? panelACompleted.value || panelBCompleted.value
      : panelACompleted.value;
  });

  // 持久化逻辑

  function createDefaultPanelSnapshot() {
    return {
      systemPrompt: DEFAULT_PROMPT.content,
      promptName: DEFAULT_PROMPT.name,
      responses: {},
    };
  }

  function applyDraftState() {
    currentSessionId.value = null;
    messages.value = [];

    panelA.responses = {};
    panelA.systemPrompt = DEFAULT_PROMPT.content;
    panelA.promptName = DEFAULT_PROMPT.name;
    panelA.error = "";
    panelA.streamingText = "";

    panelB.responses = {};
    panelB.systemPrompt = DEFAULT_PROMPT.content;
    panelB.promptName = DEFAULT_PROMPT.name;
    panelB.error = "";
    panelB.streamingText = "";

    evaluation.running = false;
    evaluation.error = "";
    evaluation.result = null;
    selectedAnnotation.index = -1;
    selectedAnnotation.field = null;
  }

  function sessionHasChat(session: ChatSession) {
    return session.messages.length > 0;
  }

  function normalizeSession(session: ChatSession): ChatSession {
    const panelAState = session.panelA ?? createDefaultPanelSnapshot();
    const panelBState = session.panelB ?? createDefaultPanelSnapshot();
    const shouldFillDefaultPrompt = session.messages.length === 0;
    const normalizePromptName = (name: string) => {
      return name === "苏格拉底引导（基础版）" ? DEFAULT_PROMPT.name : name;
    };

    return {
      ...session,
      panelA: {
        systemPrompt:
          panelAState.systemPrompt ||
          (shouldFillDefaultPrompt ? DEFAULT_PROMPT.content : ""),
        promptName: normalizePromptName(
          panelAState.promptName ||
            (shouldFillDefaultPrompt ? DEFAULT_PROMPT.name : ""),
        ),
        responses: panelAState.responses ?? {},
      },
      panelB: {
        systemPrompt:
          panelBState.systemPrompt ||
          (shouldFillDefaultPrompt ? DEFAULT_PROMPT.content : ""),
        promptName: normalizePromptName(
          panelBState.promptName ||
            (shouldFillDefaultPrompt ? DEFAULT_PROMPT.name : ""),
        ),
        responses: panelBState.responses ?? {},
      },
    };
  }

  function ensurePersistedSession() {
    // 确保当前有持久化会话：若不存在则新建并入 sessions 列表（用于保存到 localStorage）
    if (currentSessionId.value) return;

    const newSession: ChatSession = {
      id: generateId(),
      title: "新会话",
      updatedAt: Date.now(),
      mode: chatMode.value,
      messages: [],
      panelA: {
        systemPrompt: panelA.systemPrompt,
        promptName: panelA.promptName,
        responses: {},
      },
      panelB: {
        systemPrompt: panelB.systemPrompt,
        promptName: panelB.promptName,
        responses: {},
      },
      evaluation: undefined,
    };
    sessions.value.unshift(newSession);
    currentSessionId.value = newSession.id;
  }

  function normalizeProfile(profile: Partial<ModelProfile>): ModelProfile {
    return {
      id: profile.id || generateId(),
      label: profile.label || profile.name || "未命名模型",
      name: profile.name || "",
      apiKey: profile.apiKey || "",
      baseUrl: profile.baseUrl || "",
    };
  }

  function sameProfileConfig(profile: ModelProfile, model: ModelConfig) {
    return (
      profile.name === model.name &&
      profile.apiKey === model.apiKey &&
      profile.baseUrl === model.baseUrl
    );
  }

  function upsertProfileFromModel(label: string, model: ModelConfig) {
    if (!model.name && !model.apiKey && !model.baseUrl) return "";
    const existing = globalModelConfig.modelProfiles.find((profile) =>
      sameProfileConfig(profile, model),
    );
    if (existing) return existing.id;

    const profile = normalizeProfile({
      label: label || model.name || "未命名模型",
      name: model.name,
      apiKey: model.apiKey,
      baseUrl: model.baseUrl,
    });
    globalModelConfig.modelProfiles.push(profile);
    return profile.id;
  }

  function applyProfileToTarget(
    target: "panelA" | "panelB" | "evaluator",
    profileId: string,
  ) {
    const profile = globalModelConfig.modelProfiles.find(
      (item) => item.id === profileId,
    );
    if (!profile) return;
    globalModelConfig[target] = {
      name: profile.name,
      apiKey: profile.apiKey,
      baseUrl: profile.baseUrl,
    };
    globalModelConfig[`${target}ProfileId`] = profile.id;
  }

  function createModelProfile(
    profile: Omit<ModelProfile, "id"> | ModelProfile,
  ) {
    const normalized = normalizeProfile(profile);
    globalModelConfig.modelProfiles.push(normalized);
    return normalized.id;
  }

  function updateModelProfile(
    id: string,
    profile: Omit<ModelProfile, "id"> | ModelProfile,
  ) {
    const idx = globalModelConfig.modelProfiles.findIndex(
      (item) => item.id === id,
    );
    if (idx === -1) return;
    const normalized = normalizeProfile({ ...profile, id });
    globalModelConfig.modelProfiles[idx] = normalized;
    for (const target of ["panelA", "panelB", "evaluator"] as const) {
      if (globalModelConfig[`${target}ProfileId`] === id) {
        applyProfileToTarget(target, id);
      }
    }
  }

  function deleteModelProfile(id: string) {
    globalModelConfig.modelProfiles = globalModelConfig.modelProfiles.filter(
      (profile) => profile.id !== id,
    );
    for (const target of ["panelA", "panelB", "evaluator"] as const) {
      if (globalModelConfig[`${target}ProfileId`] === id) {
        globalModelConfig[`${target}ProfileId`] = "";
      }
    }
  }

  function selectModelProfile(
    target: "panelA" | "panelB" | "evaluator",
    profileId: string,
  ) {
    if (!profileId) {
      globalModelConfig[`${target}ProfileId`] = "";
      return;
    }
    applyProfileToTarget(target, profileId);
  }

  function saveConfigToStorage() {
    try {
      localStorage.setItem(
        STORAGE_KEY_CONFIG,
        JSON.stringify(globalModelConfig),
      );
    } catch (e) {
      console.warn("保存全局配置失败", e);
    }
  }

  function saveSessionsToStorage() {
    try {
      // 先更新列表中的当前会话
      if (currentSessionId.value) {
        const idx = sessions.value.findIndex(
          (s) => s.id === currentSessionId.value,
        );
        if (idx !== -1) {
          const s = sessions.value[idx];
          s.messages = [...messages.value];
          s.panelA.responses = { ...panelA.responses };
          s.panelA.systemPrompt = panelA.systemPrompt;
          s.panelA.promptName = panelA.promptName;
          s.panelB.responses = { ...panelB.responses };
          s.panelB.systemPrompt = panelB.systemPrompt;
          s.panelB.promptName = panelB.promptName;
          s.evaluation = evaluation.result ?? undefined;

          if (s.title === "新会话" && messages.value.length > 0) {
            s.title =
              messages.value[0].content.substring(0, 20) +
              (messages.value[0].content.length > 20 ? "..." : "");
          }
          s.updatedAt = Date.now();
        }
      }
      sessions.value = sessions.value.filter(sessionHasChat);
      localStorage.setItem(
        STORAGE_KEY_SESSIONS,
        JSON.stringify(sessions.value),
      );
    } catch (e) {
      console.warn("保存会话记录失败", e);
    }
  }

  function loadFromStorage() {
    try {
      const configStr = localStorage.getItem(STORAGE_KEY_CONFIG);
      if (configStr) {
        const config = JSON.parse(configStr);
        if (Array.isArray(config.modelProfiles)) {
          globalModelConfig.modelProfiles =
            config.modelProfiles.map(normalizeProfile);
        }
        if (config.panelA)
          Object.assign(globalModelConfig.panelA, config.panelA);
        if (config.panelB)
          Object.assign(globalModelConfig.panelB, config.panelB);
        if (config.evaluator)
          Object.assign(globalModelConfig.evaluator, config.evaluator);
        globalModelConfig.panelAProfileId = config.panelAProfileId || "";
        globalModelConfig.panelBProfileId = config.panelBProfileId || "";
        globalModelConfig.evaluatorProfileId = config.evaluatorProfileId || "";
        if (!globalModelConfig.panelAProfileId) {
          globalModelConfig.panelAProfileId = upsertProfileFromModel(
            "基准模型",
            globalModelConfig.panelA,
          );
        }
        if (!globalModelConfig.panelBProfileId) {
          globalModelConfig.panelBProfileId = upsertProfileFromModel(
            "对照模型",
            globalModelConfig.panelB,
          );
        }
        if (!globalModelConfig.evaluatorProfileId) {
          globalModelConfig.evaluatorProfileId = upsertProfileFromModel(
            "评测模型",
            globalModelConfig.evaluator,
          );
        }
        selectModelProfile("panelA", globalModelConfig.panelAProfileId);
        selectModelProfile("panelB", globalModelConfig.panelBProfileId);
        selectModelProfile("evaluator", globalModelConfig.evaluatorProfileId);
        if (config.generation)
          Object.assign(globalModelConfig.generation, config.generation);
      }

      const sessionsStr = localStorage.getItem(STORAGE_KEY_SESSIONS);
      if (sessionsStr) {
        sessions.value = JSON.parse(sessionsStr)
          .map(normalizeSession)
          .filter(sessionHasChat);
      }

      // 尝试切换到与初始状态匹配的会话
      if (activeModeSessions.value.length === 0) {
        createNewSession();
      } else {
        const latest = [...activeModeSessions.value].sort(
          (a, b) => b.updatedAt - a.updatedAt,
        )[0];
        switchSession(latest.id);
      }
    } catch (e) {
      console.warn("加载数据失败", e);
      createNewSession();
    }
  }

  // 会话管理

  function setChatMode(mode: ChatMode) {
    if (chatMode.value === mode) return;
    chatMode.value = mode;

    // 切换到该模式下最近的会话，若无则创建新会话
    if (activeModeSessions.value.length > 0) {
      const latest = [...activeModeSessions.value].sort(
        (a, b) => b.updatedAt - a.updatedAt,
      )[0];
      switchSession(latest.id);
    } else {
      createNewSession();
    }
  }

  function createNewSession() {
    stopStreaming();
    applyDraftState();
  }

  function switchSession(id: string) {
    stopStreaming();
    const session = sessions.value.find((s) => s.id === id);
    if (!session) return;

    currentSessionId.value = id;
    messages.value = [...session.messages];

    panelA.responses = { ...session.panelA.responses };
    panelA.systemPrompt = session.panelA.systemPrompt;
    panelA.promptName = session.panelA.promptName;
    panelA.error = "";
    panelA.streamingText = "";

    panelB.responses = { ...session.panelB.responses };
    panelB.systemPrompt = session.panelB.systemPrompt;
    panelB.promptName = session.panelB.promptName;
    panelB.error = "";
    panelB.streamingText = "";

    evaluation.running = false;
    evaluation.error = "";
    evaluation.result = session.evaluation ?? null;
    selectedAnnotation.index = session.evaluation?.annotations?.length ? 0 : -1;
    selectedAnnotation.field = null;
  }

  function deleteSession(id: string) {
    sessions.value = sessions.value.filter((s) => s.id !== id);

    if (activeModeSessions.value.length === 0) {
      createNewSession();
    } else if (currentSessionId.value === id) {
      switchSession(activeModeSessions.value[0].id);
    }
    saveSessionsToStorage();
  }

  // 自动加载
  onMounted(() => {
    // 仅加载一次
    if (sessions.value.length === 0 && !currentSessionId.value) {
      loadFromStorage();
    }
  });

  // 自动保存相关的监视器
  watch(globalModelConfig, saveConfigToStorage, { deep: true });
  watch(
    [
      messages,
      () => panelA.responses,
      () => panelA.systemPrompt,
      () => panelA.promptName,
      () => panelB.responses,
      () => panelB.systemPrompt,
      () => panelB.promptName,
      () => evaluation.result,
    ],
    () => saveSessionsToStorage(),
    { deep: true },
  );

  watch(
    () => evaluation.result?.annotations.length ?? 0,
    (annotationCount) => {
      if (annotationCount === 0) {
        selectedAnnotation.index = -1;
        selectedAnnotation.field = null;
        return;
      }
      if (
        selectedAnnotation.index < 0 ||
        selectedAnnotation.index >= annotationCount
      ) {
        selectedAnnotation.index = 0;
        selectedAnnotation.field = null;
      }
    },
  );

  // 上下文压缩与聊天逻辑

  function buildHistory(panel: PanelState, maxRounds = 10): ChatMessage[] {
    const history: ChatMessage[] = [];

    if (panel.systemPrompt) {
      history.push({ role: "system", content: panel.systemPrompt });
    }

    const totalMessages = messages.value.length;
    const startIndex = Math.max(0, totalMessages - maxRounds);

    for (let i = startIndex; i < totalMessages; i++) {
      history.push({ role: "user", content: messages.value[i].content });
      if (panel.responses[i] !== undefined) {
        history.push({ role: "assistant", content: panel.responses[i] });
      }
    }
    return history;
  }

  function streamToPanel(
    panelId: "a" | "b",
    msgIndex: number,
    temperature: number,
    maxTokens: number,
  ) {
    const panel = panelId === "a" ? panelA : panelB;
    const modelCfg =
      panelId === "a" ? globalModelConfig.panelA : globalModelConfig.panelB;

    // 基本校验：保证目标 panel 的模型配置完整
    if (!modelCfg.name || !modelCfg.apiKey || !modelCfg.baseUrl) {
      panel.error = "请配置完模型后再试";
      return;
    }

    panel.streaming = true;
    panel.streamingText = "";
    panel.error = "";

    // 通过 AbortController 支持中止流式请求
    const controller = new AbortController();
    panel.abortController = controller;

    // 构建上下文历史（压缩到最近若干轮）并发起流式请求
    const history = buildHistory(panel, 10);

    streamChat(
      modelCfg,
      history,
      {
        onChunk(chunk) {
          panel.streamingText += chunk;
        },
        onDone() {
          panel.responses[msgIndex] = panel.streamingText;
          panel.streaming = false;
          panel.streamingText = "";
          panel.abortController = null;
          if (
            panelId === "a" &&
            chatMode.value === "single" &&
            containsCompletionMarker(panel.responses[msgIndex]) &&
            hasModelConfig(globalModelConfig.evaluator)
          ) {
            void runObjectiveEvaluation();
          }
        },
        onError(err) {
          // 错误处理：记录错误信息并将已接收到的流式文本作为最终回复备份
          panel.error = err;
          if (panel.streamingText) {
            panel.responses[msgIndex] = panel.streamingText;
          }
          panel.streaming = false;
          panel.streamingText = "";
          panel.abortController = null;
        },
      },
      { temperature, maxTokens },
      controller.signal,
    );
  }

  function sendMessage(
    content: string,
    targetPanels: ("a" | "b")[] = ["a", "b"],
    temperature = globalModelConfig.generation.temperature,
    maxTokens = 2048,
  ) {
    if (!content.trim()) return;
    if (panelA.streaming || panelB.streaming) return;

    // 发送消息：保证存在持久化会话，追加 student 消息并触发各 panel 的流式生成
    ensurePersistedSession();
    const msgIndex = messages.value.length;
    messages.value.push({ role: "user", content: content.trim() });

    if (targetPanels.includes("a"))
      streamToPanel("a", msgIndex, temperature, maxTokens);
    if (targetPanels.includes("b"))
      streamToPanel("b", msgIndex, temperature, maxTokens);
  }

  function stopStreaming() {
    // 中止所有正在进行的流式请求，并把已有的流式内容保存为回复
    for (const panel of [panelA, panelB]) {
      if (panel.abortController) {
        panel.abortController.abort();
        if (panel.streamingText) {
          panel.responses[messages.value.length - 1] = panel.streamingText;
        }
        panel.streaming = false;
        panel.streamingText = "";
        panel.abortController = null;
      }
    }
  }

  function clearChat() {
    stopStreaming();
    if (currentSessionId.value) {
      sessions.value = sessions.value.filter(
        (s) => s.id !== currentSessionId.value,
      );
    }
    messages.value = [];
    evaluation.result = null;
    evaluation.error = "";
    selectedAnnotation.index = -1;
    selectedAnnotation.field = null;
    for (const panel of [panelA, panelB]) {
      panel.responses = {};
      panel.streamingText = "";
      panel.error = "";
    }
    currentSessionId.value = null;
    panelA.systemPrompt = DEFAULT_PROMPT.content;
    panelA.promptName = DEFAULT_PROMPT.name;
    panelB.systemPrompt = DEFAULT_PROMPT.content;
    panelB.promptName = DEFAULT_PROMPT.name;
    saveSessionsToStorage();
  }

  async function runObjectiveEvaluation() {
    // 触发客观评测：仅在单模型模式下可用，且必须存在至少一轮完整的师生对
    if (chatMode.value !== "single") {
      evaluation.error = "客观评测仅支持单模型评测模式";
      return;
    }
    if (evaluationDialogue.value.length < 2) {
      evaluation.error = "当前会话还没有可评测的完整师生轮次";
      return;
    }

    const sourceHash = evaluationSourceHash.value;
    const dialogue = evaluationDialogue.value;
    if (evaluation.result?.sourceHash === sourceHash) {
      evaluation.error = "";
      return;
    }

    evaluation.running = true;
    evaluation.error = "";
    try {
      evaluation.result = await evaluateDialogue(
        globalModelConfig.evaluator,
        dialogue,
        sourceHash,
      );
      selectedAnnotation.index =
        evaluation.result.annotations.length > 0 ? 0 : -1;
      selectedAnnotation.field = null;
      saveSessionsToStorage();
    } catch (err: any) {
      evaluation.error = err?.message || String(err);
    } finally {
      evaluation.running = false;
    }
  }

  function selectAnnotation(
    index: number,
    field: AnnotationFieldKey | null = null,
  ) {
    const annotations = evaluation.result?.annotations ?? [];
    if (index < 0 || index >= annotations.length) return;
    selectedAnnotation.index = index;
    selectedAnnotation.field = field;
  }

  function updateAnnotation(
    index: number,
    patch: Partial<ObjectiveAnnotation>,
  ) {
    if (!evaluation.result) return;
    const existing = evaluation.result.annotations[index];
    if (!existing) return;

    const nextAnnotations = cloneAnnotations(evaluation.result.annotations);
    nextAnnotations[index] = {
      ...existing,
      ...patch,
      speaker: existing.speaker,
      utterance: existing.utterance,
    };

    const nextMetrics = calculateObjectiveMetrics(nextAnnotations);
    if (!nextMetrics) return;

    evaluation.result = {
      ...evaluation.result,
      annotations: nextAnnotations,
      metrics: nextMetrics,
    };
  }

  function resetEditedAnnotations() {
    if (!evaluation.result) return;

    const baseline = evaluation.result.originalAnnotations?.length
      ? cloneAnnotations(evaluation.result.originalAnnotations)
      : cloneAnnotations(evaluation.result.annotations);
    const baselineMetrics = calculateObjectiveMetrics(baseline);
    if (!baselineMetrics) return;

    evaluation.result = {
      ...evaluation.result,
      annotations: baseline,
      metrics: baselineMetrics,
    };
  }

  function exportEvaluationJsonl() {
    if (!evaluation.result || evaluationStale.value) return;

    const sessionId = currentSessionId.value || evaluation.result.dialogueId;
    const sessionTitle =
      messages.value[0]?.content?.trim() || "web_demo_session";
    const payload = {
      student_id: sessionId,
      student_type: "",
      scenario: "web_demo_single_model",
      topic_id: sessionId,
      topic_text: sessionTitle,
      repeat_id: "web_demo",
      dialogue: evaluation.result.dialogue.map((turn) => ({
        role: turn.role,
        content: turn.content,
      })),
      annotations: evaluation.result.annotations,
      quality_score: evaluation.result.metrics.TotalScore.score,
    };

    downloadTextFile(
      `${JSON.stringify(payload)}\n`,
      `webdemo-annotation-${sessionId}-${buildTimestampToken(Date.now())}.jsonl`,
    );
  }

  return {
    messages,
    panelA,
    panelB,
    sendMessage,
    stopStreaming,
    clearChat,
    globalModelConfig,
    evaluation,
    evaluationDialogue,
    evaluationSourceHash,
    evaluationStale,
    canRunEvaluation,
    canExportEvaluation,
    runObjectiveEvaluation,
    selectedAnnotation,
    selectAnnotation,
    updateAnnotation,
    resetEditedAnnotations,
    exportEvaluationJsonl,
    panelACompleted,
    panelBCompleted,
    isDialogueCompleted,
    modelProfiles: computed(() => globalModelConfig.modelProfiles),
    createModelProfile,
    updateModelProfile,
    deleteModelProfile,
    selectModelProfile,
    chatMode,
    setChatMode,
    sessions,
    activeModeSessions,
    currentSessionId,
    createNewSession,
    switchSession,
    deleteSession,
  };
}
