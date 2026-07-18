import { config as loadDotEnv } from 'dotenv';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
loadDotEnv({ path: join(__dirname, '..', '..', '..', '.env'), quiet: true });

export const LLM_PROVIDERS = [
  'deepseek',
  'qwen',
  'zhipu',
  'moonshot',
  'doubao',
  'baidu',
  'custom',
] as const;

export type LlmProvider = typeof LLM_PROVIDERS[number];

interface ProviderPreset {
  displayName: string;
  apiKeyEnv: string;
  modelEnv: string;
  baseUrlEnv: string;
  defaultModel: string;
  defaultBaseUrl: string;
}

const PROVIDER_PRESETS: Record<LlmProvider, ProviderPreset> = {
  deepseek: {
    displayName: 'DeepSeek',
    apiKeyEnv: 'DEEPSEEK_API_KEY',
    modelEnv: 'DEEPSEEK_MODEL',
    baseUrlEnv: 'DEEPSEEK_BASE_URL',
    defaultModel: 'deepseek-v4-flash',
    defaultBaseUrl: 'https://api.deepseek.com',
  },
  qwen: {
    displayName: '阿里云百炼 / Qwen',
    apiKeyEnv: 'DASHSCOPE_API_KEY',
    modelEnv: 'QWEN_MODEL',
    baseUrlEnv: 'QWEN_BASE_URL',
    defaultModel: 'qwen-plus',
    defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  zhipu: {
    displayName: '智谱 GLM',
    apiKeyEnv: 'ZHIPU_API_KEY',
    modelEnv: 'ZHIPU_MODEL',
    baseUrlEnv: 'ZHIPU_BASE_URL',
    defaultModel: 'glm-5.2',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/paas/v4',
  },
  moonshot: {
    displayName: 'Kimi / Moonshot',
    apiKeyEnv: 'MOONSHOT_API_KEY',
    modelEnv: 'MOONSHOT_MODEL',
    baseUrlEnv: 'MOONSHOT_BASE_URL',
    defaultModel: 'kimi-k2.6',
    defaultBaseUrl: 'https://api.moonshot.cn/v1',
  },
  doubao: {
    displayName: '火山方舟 / 豆包',
    apiKeyEnv: 'DOUBAO_API_KEY',
    modelEnv: 'DOUBAO_MODEL',
    baseUrlEnv: 'DOUBAO_BASE_URL',
    defaultModel: '',
    defaultBaseUrl: 'https://ark.cn-beijing.volces.com/api/v3',
  },
  baidu: {
    displayName: '百度千帆 / ERNIE',
    apiKeyEnv: 'BAIDU_API_KEY',
    modelEnv: 'BAIDU_MODEL',
    baseUrlEnv: 'BAIDU_BASE_URL',
    defaultModel: 'ernie-4.5-turbo-32k',
    defaultBaseUrl: 'https://qianfan.baidubce.com/v2',
  },
  custom: {
    displayName: '自定义 OpenAI 兼容服务',
    apiKeyEnv: 'LLM_API_KEY',
    modelEnv: 'LLM_MODEL',
    baseUrlEnv: 'LLM_BASE_URL',
    defaultModel: '',
    defaultBaseUrl: '',
  },
};

export interface LlmConfig {
  provider: LlmProvider;
  displayName: string;
  apiKey: string;
  model: string;
  baseUrl: string;
  timeoutMs: number;
}

export interface PublicLlmInfo {
  provider: LlmProvider | 'none' | 'invalid';
  displayName: string;
  configured: boolean;
  model: string;
  timeoutMs?: number;
  message?: string;
}

function parseProvider(): LlmProvider | null {
  const value = process.env.LLM_PROVIDER?.trim().toLowerCase();
  if (!value) return null;
  return LLM_PROVIDERS.includes(value as LlmProvider)
    ? value as LlmProvider
    : null;
}

export function getLlmConfig(): LlmConfig {
  const rawProvider = process.env.LLM_PROVIDER?.trim().toLowerCase();
  const provider = parseProvider();

  if (!rawProvider) {
    const configuredProviders = LLM_PROVIDERS.filter(candidate => {
      const apiKeyEnv = PROVIDER_PRESETS[candidate].apiKeyEnv;
      return Boolean(process.env[apiKeyEnv]?.trim());
    });
    const suggestion = configuredProviders.length === 1
      ? `；已检测到 ${PROVIDER_PRESETS[configuredProviders[0]].displayName} 的 API Key，请在 .env 添加 LLM_PROVIDER=${configuredProviders[0]}`
      : '；请复制 .env.example 为 .env 并选择一个模型厂商';
    throw new Error(`未配置 LLM_PROVIDER${suggestion}`);
  }
  if (!provider) {
    throw new Error(`不支持的 LLM_PROVIDER=${rawProvider}；可选值：${LLM_PROVIDERS.join(', ')}`);
  }

  const preset = PROVIDER_PRESETS[provider];
  const apiKey = process.env[preset.apiKeyEnv]?.trim() ?? '';
  const model = process.env[preset.modelEnv]?.trim() || preset.defaultModel;
  const baseUrl = process.env[preset.baseUrlEnv]?.trim() || preset.defaultBaseUrl;
  const timeoutMs = Number(process.env.LLM_TIMEOUT_MS ?? 30_000);

  const missing: string[] = [];
  if (!apiKey) missing.push(preset.apiKeyEnv);
  if (!model) missing.push(preset.modelEnv);
  if (!baseUrl) missing.push(preset.baseUrlEnv);
  if (missing.length > 0) {
    throw new Error(`${preset.displayName} 配置不完整，缺少：${missing.join(', ')}`);
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error('LLM_TIMEOUT_MS 必须是大于 0 的毫秒数');
  }

  return { provider, displayName: preset.displayName, apiKey, model, baseUrl, timeoutMs };
}

export function getPublicLlmInfo(): PublicLlmInfo {
  const rawProvider = process.env.LLM_PROVIDER?.trim().toLowerCase();
  if (!rawProvider) {
    return {
      provider: 'none',
      displayName: '未配置',
      configured: false,
      model: '',
      message: '请在 .env 中配置 LLM_PROVIDER 和对应厂商的 API Key',
    };
  }

  try {
    const llm = getLlmConfig();
    return {
      provider: llm.provider,
      displayName: llm.displayName,
      configured: true,
      model: llm.model,
      timeoutMs: llm.timeoutMs,
    };
  } catch (error) {
    return {
      provider: 'invalid',
      displayName: rawProvider,
      configured: false,
      model: '',
      message: error instanceof Error ? error.message : 'LLM 配置无效',
    };
  }
}
