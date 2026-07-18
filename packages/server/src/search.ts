import { embedQuery } from './embedding.js';
import { vectorSearch, keywordSearch, normalizeChapterReferences, type SearchResult } from './vectorStore.js';

export interface HybridSearchOptions {
  topK?: number;
  vectorWeight?: number;
  keywordWeight?: number;
}

export const MAX_QUERY_LENGTH = 2_000;

export class SearchValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SearchValidationError';
  }
}

/**
 * Hybrid search: combines vector similarity with BM25 keyword matching.
 * Default weights: vector 0.7, keyword 0.3
 */
export async function hybridSearch(
  query: string,
  options: HybridSearchOptions = {}
): Promise<SearchResult[]> {
  let {
    topK = 5,
    vectorWeight = 0.7,
    keywordWeight = 0.3,
  } = options;

  if (typeof query !== 'string' || !query.trim()) {
    throw new SearchValidationError('query 必须是非空字符串');
  }
  if (query.length > MAX_QUERY_LENGTH) {
    throw new SearchValidationError(`query 不能超过 ${MAX_QUERY_LENGTH} 个字符`);
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > 20) {
    throw new SearchValidationError('topK 必须是 1 到 20 之间的整数');
  }
  if (!Number.isFinite(vectorWeight) || !Number.isFinite(keywordWeight)
      || vectorWeight < 0 || keywordWeight < 0 || vectorWeight + keywordWeight === 0) {
    throw new SearchValidationError('检索权重必须是非负数，且至少一路权重大于 0');
  }
  const weightSum = vectorWeight + keywordWeight;
  vectorWeight /= weightSum;
  keywordWeight /= weightSum;

  const normalizedQuery = normalizeChapterReferences(query.trim());
  const vectorResults = vectorWeight > 0
    ? vectorSearch(await embedQuery(normalizedQuery), topK * 2)
    : [];
  const keywordResults = keywordWeight > 0
    ? keywordSearch(normalizedQuery, topK * 2)
    : [];

  const merged = mergeResults(vectorResults, keywordResults, vectorWeight, keywordWeight);
  return merged.slice(0, topK);
}

export function mergeResults(
  vectorResults: SearchResult[],
  keywordResults: SearchResult[],
  vectorWeight: number,
  keywordWeight: number
): SearchResult[] {
  const merged = new Map<string, SearchResult>();

  const maxVectorScore = vectorResults.length > 0
    ? Math.max(...vectorResults.map(r => Math.max(0, r.score)))
    : 0;
  const maxKeywordScore = keywordResults.length > 0
    ? Math.max(...keywordResults.map(r => r.score))
    : 1;

  for (const result of vectorResults) {
    const normalizedScore = maxVectorScore > 0
      ? Math.max(0, result.score) / maxVectorScore
      : 0;
    if (normalizedScore === 0) continue;
    merged.set(result.chunk.id, {
      ...result,
      score: normalizedScore * vectorWeight,
      method: 'vector',
      vectorScore: result.score,
      keywordScore: 0,
      normalizedVectorScore: normalizedScore,
      normalizedKeywordScore: 0,
    });
  }

  for (const result of keywordResults) {
    const normalizedScore = result.score / (maxKeywordScore || 1);
    const existing = merged.get(result.chunk.id);

    if (existing) {
      existing.score += normalizedScore * keywordWeight;
      existing.method = 'hybrid';
      existing.keywordScore = result.score;
      existing.normalizedKeywordScore = normalizedScore;
    } else {
      merged.set(result.chunk.id, {
        ...result,
        score: normalizedScore * keywordWeight,
        method: 'keyword',
        vectorScore: 0,
        keywordScore: result.score,
        normalizedVectorScore: 0,
        normalizedKeywordScore: normalizedScore,
      });
    }
  }

  const results = Array.from(merged.values());
  results.sort((a, b) => b.score - a.score);
  return results.filter(result => result.score > 0);
}
