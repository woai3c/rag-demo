import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react';

interface Source {
  chunk: {
    id: string;
    content: string;
    source: string;
    heading?: string;
  };
  score: number;
  method: string;
  vectorScore?: number;
  keywordScore?: number;
  normalizedVectorScore?: number;
  normalizedKeywordScore?: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  conversational?: boolean;
  sources?: Source[];
  llm?: { provider: string; model: string };
  agent?: { steps: number; searches: number; historyMessages: number };
  weights?: { vector: number; keyword: number };
}

interface DocumentInfo {
  source: string;
  chunks: number;
  scope: 'builtin' | 'user';
}

interface Status {
  embedding: {
    provider: string;
    model: string;
    dimensions: number;
    status: 'idle' | 'loading' | 'ready' | 'error';
    expectedDownloadMB: number;
    downloadSource: string;
    cacheDirectory: string;
    progress?: number;
    progressFile?: string;
    error?: string;
  };
  llm: {
    displayName: string;
    configured: boolean;
    model: string;
    timeoutMs?: number;
    message?: string;
  };
  chunksIndexed: number;
  documents: DocumentInfo[];
}

interface Toast {
  id: number;
  type: 'success' | 'error';
  message: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState('');
  const [toast, setToast] = useState<Toast | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [mode, setMode] = useState<'ask' | 'search'>('ask');
  const [vectorWeight, setVectorWeight] = useState(0.7);
  const [topK, setTopK] = useState(5);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const statusRequestRef = useRef<Promise<void> | null>(null);

  async function refreshStatus() {
    if (statusRequestRef.current) return statusRequestRef.current;

    const request = (async () => {
      try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error('状态请求失败');
        setStatus(await response.json());
      } catch {
        setStatus(null);
      }
    })();
    statusRequestRef.current = request;
    try {
      await request;
    } finally {
      if (statusRequestRef.current === request) statusRequestRef.current = null;
    }
  }

  useEffect(() => {
    void refreshStatus();

    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') void refreshStatus();
    };
    window.addEventListener('focus', refreshWhenVisible);
    document.addEventListener('visibilitychange', refreshWhenVisible);
    return () => {
      window.removeEventListener('focus', refreshWhenVisible);
      document.removeEventListener('visibilitychange', refreshWhenVisible);
    };
  }, []);
  useEffect(() => {
    if (status?.embedding.status !== 'idle' && status?.embedding.status !== 'loading') return;

    const timer = window.setTimeout(() => void refreshStatus(), 1500);
    return () => window.clearTimeout(timer);
  }, [status]);
  useEffect(() => {
    if (!toast) return;

    const timer = window.setTimeout(() => setToast(null), 4500);
    return () => window.clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    const history = messages
      .filter(message => message.conversational)
      .slice(-8)
      .map(({ role, content }) => ({ role, content }));
    setInput('');
    setMessages(previous => [...previous, {
      role: 'user',
      content: question,
      conversational: mode === 'ask',
    }]);
    setLoading(true);
    const requestTimeoutMs = mode === 'ask'
      ? (status?.llm.timeoutMs ?? 30_000) + 5_000
      : 30_000;
    const requestController = new AbortController();
    const requestTimer = window.setTimeout(() => requestController.abort(), requestTimeoutMs);

    try {
      const endpoint = mode === 'ask' ? '/api/ask' : '/api/search';
      const payload = mode === 'ask'
        ? { question, history, topK, vectorWeight, keywordWeight: 1 - vectorWeight }
        : { query: question, topK, vectorWeight, keywordWeight: 1 - vectorWeight };
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: requestController.signal,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? '请求失败');

      if (mode === 'ask') {
        setMessages(previous => [...previous, {
          role: 'assistant',
          content: data.answer,
          conversational: true,
          sources: data.sources,
          llm: data.llm,
          agent: data.agent,
          weights: { vector: vectorWeight, keyword: 1 - vectorWeight },
        }]);
      } else {
        setMessages(previous => [...previous, {
          role: 'assistant',
          content: data.results?.length ? `找到 ${data.results.length} 条相关结果` : '未找到相关内容',
          conversational: false,
          sources: data.results,
          weights: { vector: vectorWeight, keyword: 1 - vectorWeight },
        }]);
      }
    } catch (error) {
      const errorMessage = error instanceof DOMException && error.name === 'AbortError'
        ? `${mode === 'ask' ? '生成' : '检索'}请求超时，已停止等待；请检查后端日志和模型服务`
        : error instanceof Error ? error.message : '请求失败，请确认后端已启动';
      setMessages(previous => [...previous, {
        role: 'assistant',
        content: errorMessage,
        conversational: false,
      }]);
    } finally {
      window.clearTimeout(requestTimer);
      setLoading(false);
      void refreshStatus();
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const inputElement = event.currentTarget.elements.namedItem('files') as HTMLInputElement;
    if (!inputElement.files?.length) return;

    const body = new FormData();
    Array.from(inputElement.files).forEach(file => body.append('files', file));
    setUploading(true);
    setNotice('正在解析、分块并使用本地 ONNX 模型建立索引…');

    try {
      const response = await fetch('/api/documents', { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? '上传失败');
      const summary = data.documents
        .map((document: { source: string; characters: number; chunks: number }) =>
          `${document.source.replace(/^user:/, '')}：${document.characters} 字符 / ${document.chunks} chunks`)
        .join('；');
      const successMessage = `已索引 ${data.documents.length} 个文档。${summary}`;
      setNotice(successMessage);
      setToast({ id: Date.now(), type: 'success', message: successMessage });
      inputElement.value = '';
      await refreshStatus();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '上传失败';
      setNotice(errorMessage);
      setToast({ id: Date.now(), type: 'error', message: errorMessage });
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(source: string) {
    try {
      setNotice(`正在删除 ${source.replace(/^user:/, '')}…`);
      const response = await fetch('/api/documents', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
      });
      const data = await response.json();
      setNotice(response.ok ? `已删除 ${data.removedChunks} 个 chunks` : data.error);
      await refreshStatus();
    } catch {
      setNotice('删除失败，请确认后端已启动');
    }
  }

  return (
    <div style={styles.container}>
      {toast && (
        <div
          key={toast.id}
          role={toast.type === 'error' ? 'alert' : 'status'}
          aria-live="polite"
          style={{ ...styles.toast, ...(toast.type === 'success' ? styles.toastSuccess : styles.toastError) }}
        >
          <span style={styles.toastIcon}>{toast.type === 'success' ? '✓' : '!'}</span>
          <span>{toast.message}</span>
          <button type="button" aria-label="关闭通知" style={styles.toastClose} onClick={() => setToast(null)}>×</button>
        </div>
      )}
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>RAG Demo</h1>
          <p style={styles.subtitle}>本地 ONNX 检索 + 国内大模型生成</p>
        </div>
        {status && (
          <div style={styles.statusGroup}>
            <span style={status.embedding.status === 'ready' ? styles.badgeOk : styles.badge}>
              ONNX {status.embedding.status}
              {status.embedding.status === 'loading' && status.embedding.progress !== undefined
                ? ` ${status.embedding.progress.toFixed(0)}%`
                : ''}
              {' · '}{status.chunksIndexed} chunks
            </span>
            <span style={status.llm.configured ? styles.badgeOk : styles.badgeWarn}>
              LLM · {status.llm.configured ? `${status.llm.displayName} / ${status.llm.model}` : '未配置'}
            </span>
          </div>
        )}
      </header>

      {status?.embedding.status === 'loading' && (
        <div style={styles.info}>
          首次使用正在从 {status.embedding.downloadSource} 下载约 {status.embedding.expectedDownloadMB} MB 的量化 ONNX 模型；
          缓存到 <code>{status.embedding.cacheDirectory}</code>
          {status.embedding.progressFile ? `，当前文件 ${status.embedding.progressFile}` : ''}。
        </div>
      )}
      {status && status.chunksIndexed === 0 && (
        <div style={styles.warning}>知识库为空，请先在项目根目录运行 npm run ingest，或在下方上传自己的资料。</div>
      )}

      <section style={styles.library}>
        <div style={styles.libraryHeader}>
          <div>
            <strong>资料库</strong>
            <p style={styles.hint}>支持 MD、TXT、文本型 PDF、DOCX；单文件最大 10 MB</p>
            <p style={styles.hint}>原始文件不落盘；提取后的文本分块、向量和来源保存在服务端 data/vectors.json</p>
          </div>
          <form onSubmit={handleUpload} style={styles.uploadForm} aria-busy={uploading}>
            <input name="files" type="file" multiple accept=".md,.txt,.pdf,.docx" disabled={uploading} />
            <button style={{ ...styles.smallButton, ...(uploading ? styles.buttonBusy : {}) }} disabled={uploading}>
              {uploading && <span className="spinner" aria-hidden="true" />}
              {uploading ? '正在解析并索引…' : '上传并索引'}
            </button>
          </form>
        </div>
        {notice && <p style={styles.notice}>{notice}</p>}
        <div style={styles.documents}>
          {status?.documents.map(document => (
            <span key={document.source} style={styles.documentTag}>
              {document.source.replace(/^user:/, '')} · {document.chunks}
              {document.scope === 'user' && (
                <button style={styles.deleteButton} onClick={() => void deleteDocument(document.source)}>×</button>
              )}
            </span>
          ))}
        </div>
      </section>

      <div style={styles.controls}>
        <div style={styles.modeSwitch}>
          <button style={mode === 'ask' ? styles.modeActive : styles.modeButton} onClick={() => setMode('ask')}>
            完整 RAG 问答
          </button>
          <button style={mode === 'search' ? styles.modeActive : styles.modeButton} onClick={() => setMode('search')}>
            只看混合搜索
          </button>
          <button
            type="button"
            style={styles.modeButton}
            disabled={loading || messages.length === 0}
            onClick={() => setMessages([])}
          >
            清空对话
          </button>
        </div>
        <label style={styles.weightControl}>
          向量 {vectorWeight.toFixed(1)} / BM25 {(1 - vectorWeight).toFixed(1)}
          <input type="range" min="0" max="1" step="0.1" value={vectorWeight}
            onChange={event => setVectorWeight(Number(event.target.value))} />
        </label>
        <label style={styles.weightControl}>
          Top-K
          <select value={topK} onChange={event => setTopK(Number(event.target.value))} style={styles.select}>
            {[1, 3, 5, 8, 10].map(value => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
      </div>

      {mode === 'ask' && status && !status.llm.configured && (
        <div style={styles.warning}>{status.llm.message}。混合搜索模式仍可正常使用。</div>
      )}
      {status?.embedding.status === 'error' && (
        <div style={styles.warning}>ONNX 模型加载失败：{status.embedding.error}</div>
      )}

      <main style={styles.main}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <p>{mode === 'ask' ? '提问后，系统会先检索资料，再由配置的 LLM 基于证据回答。' : '输入查询，直接观察向量、BM25 和融合排序。'}</p>
            <div style={styles.examples}>
              {['打车费超过八百块应该怎么办？', 'ERR_FIN_403 是什么意思？', '上海住宿一晚最多报多少？', '公司的年假有多少天？'].map(example => (
                <button key={example} style={styles.exampleButton} onClick={() => setInput(example)}>{example}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <article key={index} style={message.role === 'user' ? styles.userMessage : styles.assistantMessage}>
            <div style={styles.messageLabel}>
              {message.role === 'user' ? '你' : message.llm ? `${message.llm.provider} · ${message.llm.model}` : '检索结果'}
              {message.agent && (
                <span> · Agent {message.agent.steps} 步 / {message.agent.searches} 次检索 / {message.agent.historyMessages} 条历史</span>
              )}
            </div>
            <div style={styles.messageContent}>{message.content}</div>
            {message.sources && message.sources.length > 0 && (
              <details style={styles.sources}>
                <summary>证据来源与分数（{message.sources.length}）</summary>
                {message.sources.map((source, sourceIndex) => (
                  <div key={sourceIndex} style={styles.sourceCard}>
                    <div style={styles.sourceMeta}>
                      <strong>[来源 {sourceIndex + 1}] {source.chunk.source.replace(/^user:/, '')}</strong>
                      {source.chunk.heading && <span> · {source.chunk.heading}</span>}
                      <span style={styles.score}>融合分 {source.score.toFixed(3)}</span>
                    </div>
                    <div style={styles.scoreDetail}>
                      向量：原始 {source.vectorScore?.toFixed(3) ?? '—'} / 归一化 {source.normalizedVectorScore?.toFixed(3) ?? '—'}
                      {message.weights ? ` / 加权贡献 ${((source.normalizedVectorScore ?? 0) * message.weights.vector).toFixed(3)}` : ''}
                      {' · '}BM25：原始 {source.keywordScore?.toFixed(3) ?? '—'} / 归一化 {source.normalizedKeywordScore?.toFixed(3) ?? '—'}
                      {message.weights ? ` / 加权贡献 ${((source.normalizedKeywordScore ?? 0) * message.weights.keyword).toFixed(3)}` : ''}
                      {' · '}{source.method}
                    </div>
                    <p style={styles.sourceText}>{source.chunk.content}</p>
                  </div>
                ))}
              </details>
            )}
          </article>
        ))}
        {loading && (
          <div style={styles.assistantMessage} role="status" aria-live="polite">
            {mode === 'ask'
              ? `Agent 正在检索并生成…（超时上限约 ${Math.ceil((status?.llm.timeoutMs ?? 30_000) / 1000)} 秒）`
              : '正在执行混合检索…'}
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <form onSubmit={handleSubmit} style={styles.inputForm}>
        <input value={input} onChange={event => setInput(event.target.value)}
          placeholder={mode === 'ask' ? '基于资料提问…' : '输入搜索内容…'} style={styles.textInput} disabled={loading} />
        <button style={styles.sendButton} disabled={loading || !input.trim()}>发送</button>
      </form>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', minHeight: '100vh', maxWidth: 980, width: '100%', margin: '0 auto', padding: '0 18px' },
  header: { display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'center', padding: '20px 0', borderBottom: '1px solid var(--border)' },
  title: { fontSize: 24 },
  subtitle: { color: 'var(--text-muted)', fontSize: 13 },
  statusGroup: { display: 'flex', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 8 },
  badge: { padding: '6px 10px', border: '1px solid var(--border)', borderRadius: 20, fontSize: 11 },
  badgeOk: { padding: '6px 10px', border: '1px solid #14532d', color: '#86efac', borderRadius: 20, fontSize: 11 },
  badgeWarn: { padding: '6px 10px', border: '1px solid #713f12', color: '#fde68a', borderRadius: 20, fontSize: 11 },
  library: { marginTop: 14, padding: 14, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12 },
  libraryHeader: { display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'center', flexWrap: 'wrap' },
  hint: { color: 'var(--text-muted)', fontSize: 12, marginTop: 3 },
  uploadForm: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  smallButton: { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 7, minWidth: 104, background: 'var(--accent)', color: '#fff', border: 0, padding: '8px 12px', borderRadius: 8, cursor: 'pointer' },
  buttonBusy: { cursor: 'wait', opacity: 0.82 },
  notice: { color: '#a5b4fc', fontSize: 12, marginTop: 10 },
  documents: { display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 },
  documentTag: { display: 'inline-flex', alignItems: 'center', gap: 5, background: 'var(--surface-hover)', padding: '4px 8px', borderRadius: 6, fontSize: 11 },
  deleteButton: { border: 0, background: 'transparent', color: '#fca5a5', cursor: 'pointer', fontSize: 16, lineHeight: 1 },
  controls: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, padding: '12px 0', flexWrap: 'wrap' },
  modeSwitch: { display: 'flex', gap: 5 },
  modeButton: { padding: '7px 13px', border: '1px solid var(--border)', borderRadius: 8, background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer' },
  modeActive: { padding: '7px 13px', border: '1px solid var(--accent)', borderRadius: 8, background: 'var(--accent)', color: '#fff', cursor: 'pointer' },
  weightControl: { display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 },
  select: { padding: '5px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)' },
  warning: { padding: 10, border: '1px solid #713f12', background: '#42200655', color: '#fde68a', borderRadius: 8, fontSize: 12 },
  info: { marginTop: 12, padding: 10, border: '1px solid #1e3a8a', background: '#17255455', color: '#bfdbfe', borderRadius: 8, fontSize: 12 },
  main: { flex: 1, minHeight: 300, padding: '12px 0' },
  empty: { minHeight: 280, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', gap: 18 },
  examples: { display: 'flex', justifyContent: 'center', gap: 8, flexWrap: 'wrap' },
  exampleButton: { padding: '8px 12px', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--surface)', color: 'var(--text)', cursor: 'pointer' },
  userMessage: { marginBottom: 12, padding: 14, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12 },
  assistantMessage: { marginBottom: 12, padding: 14, border: '1px solid var(--border)', borderRadius: 12 },
  messageLabel: { color: 'var(--text-muted)', fontSize: 11, marginBottom: 6 },
  messageContent: { whiteSpace: 'pre-wrap', lineHeight: 1.75 },
  sources: { marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', color: '#a5b4fc', fontSize: 12 },
  sourceCard: { marginTop: 8, padding: 10, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' },
  sourceMeta: { display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center', fontSize: 11 },
  score: { marginLeft: 'auto', color: '#a5b4fc' },
  scoreDetail: { color: 'var(--text-muted)', fontSize: 10, marginTop: 4 },
  sourceText: { color: 'var(--text-muted)', fontSize: 12, marginTop: 6, whiteSpace: 'pre-wrap', maxHeight: 150, overflow: 'auto' },
  inputForm: { position: 'sticky', bottom: 0, display: 'flex', gap: 8, padding: '14px 0', background: 'var(--bg)', borderTop: '1px solid var(--border)' },
  textInput: { flex: 1, padding: '12px 14px', borderRadius: 10, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)', outline: 'none' },
  sendButton: { padding: '12px 22px', border: 0, borderRadius: 10, background: 'var(--accent)', color: '#fff', cursor: 'pointer' },
  toast: { position: 'fixed', zIndex: 1000, top: 18, right: 18, display: 'flex', alignItems: 'flex-start', gap: 10, width: 'min(430px, calc(100vw - 36px))', padding: '12px 14px', borderRadius: 10, boxShadow: '0 14px 35px #0008', fontSize: 13 },
  toastSuccess: { border: '1px solid #166534', background: '#052e16', color: '#dcfce7' },
  toastError: { border: '1px solid #991b1b', background: '#450a0a', color: '#fee2e2' },
  toastIcon: { display: 'inline-flex', flex: '0 0 auto', alignItems: 'center', justifyContent: 'center', width: 20, height: 20, borderRadius: '50%', border: '1px solid currentColor', fontWeight: 700, lineHeight: 1 },
  toastClose: { flex: '0 0 auto', marginLeft: 'auto', border: 0, background: 'transparent', color: 'inherit', cursor: 'pointer', fontSize: 20, lineHeight: 1 },
};
