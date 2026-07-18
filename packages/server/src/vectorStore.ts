import { readFileSync, writeFileSync, existsSync, mkdirSync, renameSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Chunk } from './chunking.js';

const CHINESE_DIGITS: Record<string, number> = {
  一: 1,
  二: 2,
  两: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
};

function parseChineseNumber(value: string): number | null {
  if (value === '十') return 10;
  const tenIndex = value.indexOf('十');
  if (tenIndex >= 0) {
    const tens = tenIndex === 0 ? 1 : CHINESE_DIGITS[value.slice(0, tenIndex)];
    const unitsText = value.slice(tenIndex + 1);
    const units = unitsText === '' ? 0 : CHINESE_DIGITS[unitsText];
    return tens !== undefined && units !== undefined ? tens * 10 + units : null;
  }
  return CHINESE_DIGITS[value] ?? null;
}

/** 将“第一章/第二十章”等写法统一为“第1章/第20章”，提高章节号检索的一致性。 */
export function normalizeChapterReferences(text: string): string {
  return text.replace(/第([一二两三四五六七八九十]{1,3})章/g, (match, numberText: string) => {
    const number = parseChineseNumber(numberText);
    return number === null ? match : `第${number}章`;
  });
}

export interface StoredChunk extends Chunk {
  vector: number[];
}

export interface SearchResult {
  chunk: Chunk;
  score: number;
  method: 'vector' | 'keyword' | 'hybrid';
  vectorScore?: number;
  keywordScore?: number;
  normalizedVectorScore?: number;
  normalizedKeywordScore?: number;
}

interface StoreFile {
  version: 2;
  embeddingModel: string;
  chunks: StoredChunk[];
}

export interface DocumentInfo {
  source: string;
  chunks: number;
  scope: 'builtin' | 'user';
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = join(__dirname, '..', '..', '..');
const DATA_DIR = join(ROOT_DIR, 'data');
const STORE_PATH = join(DATA_DIR, 'vectors.json');

let store: StoredChunk[] = [];
let currentEmbeddingModel = '';

export function loadStore(expectedEmbeddingModel: string): void {
  if (existsSync(STORE_PATH)) {
    let parsed: StoreFile | StoredChunk[];
    try {
      parsed = JSON.parse(readFileSync(STORE_PATH, 'utf-8')) as StoreFile | StoredChunk[];
    } catch (error) {
      store = [];
      currentEmbeddingModel = expectedEmbeddingModel;
      console.warn('[VectorStore] 索引文件无法解析，将使用空索引；请重新运行 npm run ingest', error);
      return;
    }

    if (Array.isArray(parsed)) {
      store = [];
      currentEmbeddingModel = expectedEmbeddingModel;
      console.warn('[VectorStore] 检测到旧版索引格式，请运行 npm run ingest 重新建立 ONNX 索引');
      return;
    }
    if (!parsed || parsed.version !== 2 || !Array.isArray(parsed.chunks)) {
      store = [];
      currentEmbeddingModel = expectedEmbeddingModel;
      console.warn('[VectorStore] 索引结构无效，将使用空索引；请重新运行 npm run ingest');
      return;
    }
    if (parsed.embeddingModel !== expectedEmbeddingModel) {
      store = [];
      currentEmbeddingModel = expectedEmbeddingModel;
      console.warn(`[VectorStore] 索引模型为 ${parsed.embeddingModel}，当前模型为 ${expectedEmbeddingModel}；请重新运行 npm run ingest`);
      return;
    }

    store = parsed.chunks;
    currentEmbeddingModel = parsed.embeddingModel;
    console.log(`[VectorStore] 已加载 ${store.length} 个 chunks`);
  } else {
    store = [];
    currentEmbeddingModel = expectedEmbeddingModel;
    console.log('[VectorStore] 空存储，请先运行 npm run ingest');
  }
}

export function saveStore(embeddingModel: string = currentEmbeddingModel): void {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }
  currentEmbeddingModel = embeddingModel;
  const data: StoreFile = {
    version: 2,
    embeddingModel,
    chunks: store,
  };
  const temporaryPath = `${STORE_PATH}.tmp`;
  writeFileSync(temporaryPath, JSON.stringify(data, null, 2));
  renameSync(temporaryPath, STORE_PATH);
  console.log(`[VectorStore] 已保存 ${store.length} 个 chunks`);
}

export function addChunks(chunks: StoredChunk[]): void {
  const existingIds = new Set(store.map(c => c.id));
  const newChunks = chunks.filter(c => !existingIds.has(c.id));
  store.push(...newChunks);
  console.log(`[VectorStore] 新增 ${newChunks.length} 个 chunks（跳过 ${chunks.length - newChunks.length} 个重复）`);
}

export function clearStore(): void {
  store = [];
}

export function removeBuiltInChunks(): number {
  const before = store.length;
  store = store.filter(chunk => chunk.source.startsWith('user:'));
  return before - store.length;
}

export function getStoreSize(): number {
  return store.length;
}

export function removeBySource(source: string): number {
  const before = store.length;
  store = store.filter(chunk => chunk.source !== source);
  return before - store.length;
}

export function listDocuments(): DocumentInfo[] {
  const counts = new Map<string, number>();
  for (const chunk of store) {
    counts.set(chunk.source, (counts.get(chunk.source) ?? 0) + 1);
  }
  return Array.from(counts, ([source, chunks]) => ({
    source,
    chunks,
    scope: source.startsWith('user:') ? 'user' as const : 'builtin' as const,
  })).sort((a, b) => a.source.localeCompare(b.source));
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dotProduct / denominator;
}

export function vectorSearch(queryVector: number[], topK: number = 5): SearchResult[] {
  if (queryVector.length === 0 || store.length === 0) return [];

  const scored = store.map(chunk => ({
    chunk: {
      id: chunk.id,
      content: chunk.content,
      source: chunk.source,
      heading: chunk.heading,
      startOffset: chunk.startOffset,
    },
    score: cosineSimilarity(queryVector, chunk.vector),
    method: 'vector' as const,
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

/**
 * BM25-inspired keyword search.
 * Simplified implementation suitable for demo purposes.
 */
export function keywordSearch(query: string, topK: number = 5): SearchResult[] {
  const terms = Array.from(new Set(tokenize(query)));
  if (terms.length === 0 || store.length === 0) return [];

  const N = store.length;
  const tokenizedDocuments = store.map(chunk => tokenize(normalizeChapterReferences(chunk.content)));
  const avgDl = tokenizedDocuments.reduce((sum, tokens) => sum + tokens.length, 0) / N;
  const k1 = 1.2;
  const b = 0.75;

  const df = new Map<string, number>();
  for (const term of terms) {
    let count = 0;
    for (const documentTokens of tokenizedDocuments) {
      if (documentTokens.includes(term)) count++;
    }
    df.set(term, count);
  }

  const scored = store.map((chunk, index) => {
    const documentTokens = tokenizedDocuments[index];
    const frequencies = countTerms(documentTokens);
    const dl = documentTokens.length;
    let score = 0;

    for (const term of terms) {
      const tf = frequencies.get(term) ?? 0;
      if (tf === 0) continue;

      const docFreq = df.get(term) ?? 0;
      const idf = Math.log((N - docFreq + 0.5) / (docFreq + 0.5) + 1);
      const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgDl));
      score += idf * tfNorm;
    }

    return {
      chunk: {
        id: chunk.id,
        content: chunk.content,
        source: chunk.source,
        heading: chunk.heading,
        startOffset: chunk.startOffset,
      },
      score,
      method: 'keyword' as const,
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.filter(r => r.score > 0).slice(0, topK);
}

export function tokenize(text: string): string[] {
  const segments = text.toLowerCase().match(/[a-z0-9_]+|[\u4e00-\u9fff]+/g) ?? [];
  const tokens: string[] = [];

  for (const segment of segments) {
    if (!/[\u4e00-\u9fff]/.test(segment)) {
      tokens.push(segment);
      continue;
    }

    if (segment.length === 1) {
      tokens.push(segment);
      continue;
    }
    for (let index = 0; index < segment.length - 1; index++) {
      tokens.push(segment.slice(index, index + 2));
    }
  }

  return tokens;
}

function countTerms(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return counts;
}
