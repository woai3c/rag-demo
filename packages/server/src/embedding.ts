import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, statSync, unlinkSync } from 'fs';

export const EMBEDDING_MODEL = 'Xenova/multilingual-e5-small';
export const EMBEDDING_REVISION = '761b726dd34fb83930e26aab4e9ac3899aa1fa78';
export const EMBEDDING_INDEX_ID = `${EMBEDDING_MODEL}@${EMBEDDING_REVISION}:q8`;
export const EMBEDDING_DIMENSIONS = 384;
export const EMBEDDING_EXPECTED_DOWNLOAD_MB = 135;

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = join(__dirname, '..', '..', '..');
const DEFAULT_CACHE_DIR = join(ROOT_DIR, '.cache', 'transformers');
const EXPECTED_CACHE_FILES: Record<string, number> = {
  'config.json': 658,
  'tokenizer.json': 17_082_730,
  'tokenizer_config.json': 443,
  'onnx/model_quantized.onnx': 118_308_185,
};

let transformersPipeline: any = null;
let initializing: Promise<any> | null = null;
let embeddingStatus: 'idle' | 'loading' | 'ready' | 'error' = 'idle';
let embeddingError = '';
let embeddingProgress: number | undefined;
let embeddingProgressFile = '';
let lastReportedFile = '';
let lastReportedBucket = -1;

interface DownloadProgressEvent {
  status?: string;
  file?: string;
  progress?: number;
}

function handleDownloadProgress(event: DownloadProgressEvent): void {
  if (event.status === 'progress' && Number.isFinite(event.progress)) {
    embeddingProgress = Math.max(0, Math.min(100, event.progress as number));
    embeddingProgressFile = event.file ?? '';
    const bucket = Math.floor(embeddingProgress / 10) * 10;
    if (embeddingProgressFile !== lastReportedFile || bucket !== lastReportedBucket) {
      console.log(`[Embedding] 下载 ${embeddingProgressFile || '模型文件'}：${embeddingProgress.toFixed(0)}%`);
      lastReportedFile = embeddingProgressFile;
      lastReportedBucket = bucket;
    }
  }
}

function removeIncompleteCacheFiles(cacheDirectory: string): void {
  for (const [relativePath, expectedBytes] of Object.entries(EXPECTED_CACHE_FILES)) {
    const filePath = join(cacheDirectory, EMBEDDING_MODEL, relativePath);
    if (!existsSync(filePath)) continue;
    const actualBytes = statSync(filePath).size;
    if (actualBytes === expectedBytes) continue;
    unlinkSync(filePath);
    console.warn(`[Embedding] 删除未完整下载的缓存文件 ${relativePath}（${actualBytes}/${expectedBytes} bytes）`);
  }
}

export async function initializeEmbedding(): Promise<void> {
  if (transformersPipeline) return;
  if (!initializing) {
    embeddingStatus = 'loading';
    embeddingError = '';
    initializing = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.cacheDir = process.env.EMBEDDING_CACHE_DIR?.trim() || DEFAULT_CACHE_DIR;
      removeIncompleteCacheFiles(env.cacheDir);
      const instance = await pipeline('feature-extraction', EMBEDDING_MODEL, {
        dtype: 'q8',
        revision: EMBEDDING_REVISION,
        progress_callback: handleDownloadProgress,
      });
      console.log(`[Embedding] 使用本地 ONNX 模型 ${EMBEDDING_MODEL}，缓存目录 ${env.cacheDir}`);
      return instance;
    })();
  }

  try {
    transformersPipeline = await initializing;
    embeddingStatus = 'ready';
    embeddingProgress = 100;
    embeddingProgressFile = '';
  } catch (error) {
    embeddingStatus = 'error';
    embeddingError = error instanceof Error ? error.message : 'ONNX 模型加载失败';
    throw error;
  } finally {
    initializing = null;
  }
}

async function embedPrepared(text: string): Promise<number[]> {
  await initializeEmbedding();
  const output = await transformersPipeline(text, {
    pooling: 'mean',
    normalize: true,
  });
  return Array.from(output.data as Float32Array);
}

export async function embedQuery(text: string): Promise<number[]> {
  return embedPrepared(`query: ${text.trim()}`);
}

export async function embedPassage(text: string): Promise<number[]> {
  return embedPrepared(`passage: ${text.trim()}`);
}

export async function embedPassages(texts: string[]): Promise<number[][]> {
  const vectors: number[][] = [];
  for (const text of texts) {
    vectors.push(await embedPassage(text));
  }
  return vectors;
}

export function getEmbeddingInfo(): {
  provider: 'onnx';
  model: string;
  revision: string;
  dimensions: number;
  status: 'idle' | 'loading' | 'ready' | 'error';
  expectedDownloadMB: number;
  downloadSource: string;
  cacheDirectory: string;
  progress?: number;
  progressFile?: string;
  error?: string;
} {
  return {
    provider: 'onnx',
    model: EMBEDDING_MODEL,
    revision: EMBEDDING_REVISION,
    dimensions: EMBEDDING_DIMENSIONS,
    status: embeddingStatus,
    expectedDownloadMB: EMBEDDING_EXPECTED_DOWNLOAD_MB,
    downloadSource: 'Hugging Face Hub',
    cacheDirectory: process.env.EMBEDDING_CACHE_DIR?.trim() || '.cache/transformers',
    ...(embeddingProgress !== undefined ? { progress: embeddingProgress } : {}),
    ...(embeddingProgressFile ? { progressFile: embeddingProgressFile } : {}),
    ...(embeddingError ? { error: embeddingError } : {}),
  };
}
