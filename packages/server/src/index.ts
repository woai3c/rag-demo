import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { randomUUID } from 'crypto';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { loadStore, getStoreSize, listDocuments, removeBySource, saveStore } from './vectorStore.js';
import { initializeEmbedding, getEmbeddingInfo, EMBEDDING_INDEX_ID } from './embedding.js';
import { getPublicLlmInfo } from './config.js';
import { hybridSearch, MAX_QUERY_LENGTH, SearchValidationError } from './search.js';
import { ragQuery, type ConversationMessage } from './rag.js';
import { ingestUploadedFiles, isSupportedDocument } from './documents.js';

const app = express();
const PORT = process.env.PORT ?? 3001;
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_LENGTH = 4_000;
const MAX_HISTORY_TOTAL_LENGTH = 16_000;

function parseConversationHistory(value: unknown): ConversationMessage[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new SearchValidationError('history 必须是消息数组');
  if (value.length > MAX_HISTORY_MESSAGES) {
    throw new SearchValidationError(`history 最多包含 ${MAX_HISTORY_MESSAGES} 条消息`);
  }

  let totalLength = 0;
  return value.map((item, index) => {
    if (!item || typeof item !== 'object') {
      throw new SearchValidationError(`history[${index}] 格式无效`);
    }
    const role = 'role' in item ? item.role : undefined;
    const content = 'content' in item ? item.content : undefined;
    if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string' || !content.trim()) {
      throw new SearchValidationError(`history[${index}] 必须包含有效的 role 和 content`);
    }
    if (content.length > MAX_HISTORY_MESSAGE_LENGTH) {
      throw new SearchValidationError(`history[${index}].content 不能超过 ${MAX_HISTORY_MESSAGE_LENGTH} 个字符`);
    }
    totalLength += content.length;
    if (totalLength > MAX_HISTORY_TOTAL_LENGTH) {
      throw new SearchValidationError(`history 总长度不能超过 ${MAX_HISTORY_TOTAL_LENGTH} 个字符`);
    }
    return { role, content: content.trim() };
  });
}

app.use(cors());
app.use(express.json({ limit: '1mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    files: 10,
    fileSize: 10 * 1024 * 1024,
  },
  fileFilter: (_req, file, callback) => {
    if (isSupportedDocument(file.originalname)) {
      callback(null, true);
    } else {
      callback(new Error('仅支持 .md、.txt、文本型 .pdf 和 .docx'));
    }
  },
});

app.get('/api/status', (_req, res) => {
  const embedding = getEmbeddingInfo();
  res.json({
    status: 'ok',
    embedding,
    llm: getPublicLlmInfo(),
    chunksIndexed: getStoreSize(),
    documents: listDocuments(),
  });
});

app.post('/api/documents', upload.array('files', 10), async (req, res) => {
  try {
    const files = req.files as Express.Multer.File[] | undefined;
    if (!files || files.length === 0) {
      res.status(400).json({ error: '请选择至少一个文档' });
      return;
    }

    const documents = await ingestUploadedFiles(files);
    res.status(201).json({ documents, chunksIndexed: getStoreSize() });
  } catch (error) {
    console.error('[Document Upload Error]', error);
    res.status(400).json({ error: error instanceof Error ? error.message : '文档处理失败' });
  }
});

app.delete('/api/documents', (req, res) => {
  const source = typeof req.body?.source === 'string' ? req.body.source : '';
  if (!source.startsWith('user:')) {
    res.status(400).json({ error: '只能删除用户上传的文档' });
    return;
  }

  const removedChunks = removeBySource(source);
  saveStore(EMBEDDING_INDEX_ID);
  res.json({ removedChunks, chunksIndexed: getStoreSize() });
});

app.post('/api/search', async (req, res) => {
  try {
    const { query, topK = 5, vectorWeight = 0.7, keywordWeight = 0.3 } = req.body;
    if (typeof query !== 'string' || !query.trim()) {
      res.status(400).json({ error: 'query 必须是非空字符串' });
      return;
    }
    if (query.length > MAX_QUERY_LENGTH) {
      res.status(400).json({ error: `query 不能超过 ${MAX_QUERY_LENGTH} 个字符` });
      return;
    }
    if (getStoreSize() === 0) {
      res.status(409).json({ error: '知识库为空，请先运行 npm run ingest 或上传文档' });
      return;
    }

    const results = await hybridSearch(query, { topK, vectorWeight, keywordWeight });
    res.json({ results });
  } catch (error) {
    console.error('[Search Error]', error);
    const embedding = getEmbeddingInfo();
    const status = error instanceof SearchValidationError
      ? 400
      : embedding.status === 'error' ? 503 : 500;
    res.status(status).json({
      error: error instanceof Error ? error.message : '搜索失败',
    });
  }
});

app.post('/api/ask', async (req, res) => {
  const requestId = randomUUID().slice(0, 8);
  const startedAt = Date.now();
  const requestController = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) requestController.abort();
  });
  try {
    const { question, history: rawHistory, topK = 5, vectorWeight = 0.7, keywordWeight = 0.3 } = req.body;
    if (typeof question !== 'string' || !question.trim()) {
      res.status(400).json({ error: 'question 必须是非空字符串' });
      return;
    }
    if (question.length > MAX_QUERY_LENGTH) {
      res.status(400).json({ error: `question 不能超过 ${MAX_QUERY_LENGTH} 个字符` });
      return;
    }
    if (getStoreSize() === 0) {
      res.status(409).json({ error: '知识库为空，请先运行 npm run ingest 或上传文档' });
      return;
    }

    const history = parseConversationHistory(rawHistory);
    console.log(`[RAG ${requestId}] 收到问题：${question.trim().slice(0, 80)}（历史消息 ${history.length} 条）`);
    const response = await ragQuery(question, {
      topK,
      searchOptions: { vectorWeight, keywordWeight },
      requestId,
      history,
      signal: requestController.signal,
    });
    console.log(`[RAG ${requestId}] 请求完成，总耗时 ${Date.now() - startedAt}ms`);
    res.json(response);
  } catch (error) {
    console.error(`[RAG ${requestId}] 请求失败，总耗时 ${Date.now() - startedAt}ms`, error);
    const llm = getPublicLlmInfo();
    const embedding = getEmbeddingInfo();
    const status = error instanceof SearchValidationError
      ? 400
      : !llm.configured || embedding.status === 'error' ? 503 : 502;
    res.status(status).json({
      error: error instanceof Error ? error.message : 'RAG 查询失败',
    });
  }
});

export function startServer() {
  console.log('=== RAG Demo Server ===\n');

  loadStore(EMBEDDING_INDEX_ID);

  const { provider, model, dimensions } = getEmbeddingInfo();
  console.log(`\nEmbedding: ${provider} / ${model} (${dimensions}维)`);
  console.log(`Chunks: ${getStoreSize()}`);
  const llm = getPublicLlmInfo();
  console.log(`LLM: ${llm.configured ? `${llm.displayName} / ${llm.model}` : llm.message}`);

  const server = app.listen(PORT, () => {
    console.log(`\nServer running at http://localhost:${PORT}`);
  });

  void initializeEmbedding().catch(error => {
    console.error('[Embedding Init Error]', error);
  });
  return server;
}

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('[HTTP Error]', error);
  const message = error instanceof multer.MulterError
    ? `上传失败：${error.message}`
    : error instanceof Error ? error.message : '请求处理失败';
  res.status(400).json({ error: message });
});

export { app };

const entryPath = process.argv[1] ? resolve(process.argv[1]) : '';
if (entryPath === fileURLToPath(import.meta.url)) {
  startServer();
}
