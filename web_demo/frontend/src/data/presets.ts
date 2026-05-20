export interface PresetPrompt {
  id: string
  name: string
  description: string
  content: string
}

export const presetPrompts: PresetPrompt[] = [
  {
    id: 'socratic_base',
    name: '苏格拉底引导',
    description: '通过逐轮递进的问题引导学生独立思考，鼓励跨学科联系',
    content: `你是一位跨学科的教师，始终使用苏格拉底式提问法来引导学生。
目标：通过逐轮递进的问题，引导学生独立思考。

规范：
1. 每一轮只能提出**一个**简洁的问题。
2. 基于学生的回答提炼关键点，推进思考。
3. 鼓励跨学科思考（生物、地理、物理、历史等）。
4. 不要直接给答案，除非学生完全卡住需要提供脚手架。
5. 当判断教学目标达成时，必须先用一句话概括学生已经掌握的关键结论，再**严格以字符串 [结束] 结尾**；禁止只输出 [结束]。`,
  },
  {
    id: 'custom',
    name: '自定义 Prompt',
    description: '自由编写你自己的 System Prompt',
    content: '',
  },
]

export interface QuickModel {
  name: string
  baseUrl: string
  label: string
}

export const defaultModels: QuickModel[] = [
  {
    name: 'qwen-plus',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    label: 'Qwen Plus',
  },
  {
    name: 'qwen-turbo',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    label: 'Qwen Turbo',
  },
  {
    name: 'qwen-max',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    label: 'Qwen Max',
  },
  {
    name: 'gpt-4o-mini',
    baseUrl: 'https://api.openai.com/v1',
    label: 'GPT-4o Mini',
  },
  {
    name: 'gpt-4o',
    baseUrl: 'https://api.openai.com/v1',
    label: 'GPT-4o',
  },
]
