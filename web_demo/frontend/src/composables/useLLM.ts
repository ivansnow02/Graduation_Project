/**
 * useLLM - 直接使用 OpenAI SDK 调用 LLM API（支持 vLLM / DashScope 等兼容端点）
 */
import OpenAI from 'openai'

export interface ModelConfig {
  name: string
  apiKey: string
  baseUrl: string
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface StreamCallbacks {
  onChunk: (chunk: string) => void
  onDone: () => void
  onError: (error: string) => void
}

export function useLLM() {
  /**
   * 创建 OpenAI 客户端（支持任意 OpenAI 兼容端点）
   */
  function createClient(model: ModelConfig): OpenAI {
    return new OpenAI({
      apiKey: model.apiKey,
      baseURL: model.baseUrl,
      dangerouslyAllowBrowser: true, // 演示系统，允许浏览器端使用
    })
  }

  /**
   * 发起流式对话请求
   */
  async function streamChat(
    model: ModelConfig,
    messages: ChatMessage[],
    { onChunk, onDone, onError }: StreamCallbacks,
    options?: { temperature?: number; maxTokens?: number },
    signal?: AbortSignal
  ): Promise<void> {
    try {
      const client = createClient(model)

      const stream = await client.chat.completions.create(
        {
          model: model.name,
          messages,
          stream: true,
          temperature: options?.temperature ?? 0.8,
          max_tokens: options?.maxTokens ?? 2048,
        },
        { signal }
      )

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content
        if (delta) {
          onChunk(delta)
        }
        if (chunk.choices[0]?.finish_reason === 'stop') {
          break
        }
      }
      onDone()
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      onError(err?.message || String(err))
    }
  }

  return { streamChat }
}
