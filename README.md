# RAG Demo

一个完整的 RAG（Retrieval-Augmented Generation，检索增强生成）教学项目：服务端使用本地 ONNX 模型完成文档向量化，以向量检索和 BM25 进行混合召回，再调用配置的大语言模型生成带来源的回答。

## 实现能力

- 本地 ONNX Embedding：`Xenova/multilingual-e5-small`，384 维，q8 量化权重，并固定模型 revision
- 混合检索：余弦相似度向量检索 + BM25 关键词检索
- 分数解释：展示两路原始分、归一化分、加权贡献和最终融合分
- 完整 RAG：Top-K 片段组装、LLM 生成、来源展示
- Markdown 回答：使用 `react-markdown` 和 `remark-gfm` 渲染标题、列表、表格、引用与代码块
- 国内模型预设：DeepSeek、阿里云百炼/Qwen、智谱 GLM、Kimi/Moonshot、火山方舟/豆包、百度千帆
- 自定义 OpenAI-compatible 服务
- 内置资料均为 mock 数据，首次索引后即可复现示例问题
- 网页上传 `.md`、`.txt`、文本型 `.pdf`、`.docx`，并支持删除用户资料

## 运行环境

- 建议使用 Node.js 22 LTS
- npm 10 或兼容版本
- 首次建立索引时需要能够访问 `huggingface.co`
- 完整 RAG 问答需要一个受支持厂商的 API Key

## 首次使用

### 1. 安装依赖

在项目根目录执行：

```bash
npm install
```

这一步安装 Node.js 依赖，但不会把 Embedding 模型打包进仓库。

### 2. 配置生成模型

PowerShell：

```powershell
Copy-Item .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

编辑 `.env`：

1. 设置 `LLM_PROVIDER`。
2. 填写这个厂商对应的 API Key。
3. 如果账号、地域或控制台要求不同的模型 ID，再覆盖相应的 `*_MODEL` 或 `*_BASE_URL`。
4. `LLM_TIMEOUT_MS` 是整次 Agent Loop 的超时上限，不是每次请求的实际耗时；默认 30 秒。

例如使用 DeepSeek：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的_API_Key
```

API Key 只由 Node.js 服务端读取，不会返回浏览器。

### 3. 下载 ONNX 模型并建立内置索引

```bash
npm run ingest
```

第一次运行会发生以下事情：

1. `@huggingface/transformers` 从 Hugging Face Hub 下载 `Xenova/multilingual-e5-small`。
2. 本项目使用 q8 量化 ONNX 权重，并固定到 revision `761b726dd34fb83930e26aab4e9ac3899aa1fa78`。模型权重约 118 MB，加上 tokenizer 和配置文件，总下载量约 135 MB。
3. 文件默认缓存在项目根目录的 `.cache/transformers/`，后续索引和启动会复用缓存。
4. 服务端将 `seedDocuments.ts` 中的内置示例资料分块、向量化，并写入 `data/vectors.json`。

模型由 Node.js 服务端下载和执行，不是由浏览器下载。下载耗时取决于网络速度；终端会显示下载进度。模型文件列表和大小可在 [固定 revision 的 Hugging Face 模型仓库](https://huggingface.co/Xenova/multilingual-e5-small/tree/761b726dd34fb83930e26aab4e9ac3899aa1fa78/onnx) 核对。

如果下载中断或出现 `ConnectTimeoutError`，确认当前网络可以访问 `https://huggingface.co` 后重新运行 `npm run ingest`。初始化程序会校验固定 revision 的缓存文件大小，自动删除未完整下载的文件，避免把残缺权重当成有效缓存。缓存目录可以在 `.env` 中覆盖：

```dotenv
EMBEDDING_CACHE_DIR=D:\model-cache\transformers
```

不要提交模型缓存；`.cache/` 已加入 `.gitignore`。

### 4. 启动网页和服务端

```bash
npm run dev
```

启动后访问：

- 网页端：<http://localhost:5173>
- 服务端状态：<http://localhost:3001/api/status>

页面右上角应显示：

- `ONNX ready`：Embedding 模型已加载
- 大于 0 的 `chunks`：知识库已经建立索引
- LLM 厂商和模型名：`.env` 配置有效

如果直接运行 `npm run dev` 而没有先执行 `npm run ingest`，服务端仍会初始化并缓存 ONNX 模型，但知识库是空的；页面会提示先建立内置索引或上传资料。

## 如何体验完整 RAG

首次建议依次点击页面中的四个问题：

1. `打车费超过八百块应该怎么办？`——观察语义召回。
2. `ERR_FIN_403 是什么意思？`——观察 BM25 对精确错误码的作用。
3. `上海住宿一晚最多报多少？`——核对数字和附加条件。
4. `公司的年假有多少天？`——检查资料不足时是否拒绝编造。

在“完整 RAG 问答”中，系统会：

```text
当前问题 + 最近的会话历史
  → 把“它/这个问题”等追问补成可独立检索的问题
  → 本地 ONNX 生成查询向量
  → 向量检索 + BM25
  → 分数归一化和加权融合
  → 选取初始 Top-K 证据
  → 调用 .env 选择的 LLM 进入 Agent Loop
  → 证据不足时由模型调用知识库搜索工具再次检索
  → 证据足够或达到循环上限时生成最终回答
  → 返回答案、来源和检索分数
```

因此可以连续追问，例如先问“第一章有什么问题？”，再问“那怎么修复这个问题？”。浏览器会把当前页面内最近 8 条有效问答随请求发送给服务端；服务端不会在数据库中另存聊天记录，刷新页面后会话历史会清空。

Agent Loop 保持有界：最多执行 3 个 LLM 步骤，每轮最多接受 2 个工具调用，只开放只读的 `search_knowledge_base`，并拒绝未知工具、无效参数和重复查询。整轮最多汇总 20 个去重来源。

页面上的“只看混合搜索”是检索诊断视图，用于比较权重和检查正确片段是否进入 Top-K；完整问答流程使用“完整 RAG 问答”。

可以调整：

- 向量/BM25 权重：`1.0/0.0`、`0.0/1.0` 或混合比例
- Top-K：最终进入上下文的证据数量

融合分是排序分数，不是“答案正确率”或概率。

## 使用自己的资料

在网页“资料库”区域选择文件并点击“上传并索引”。

支持范围：

| 项目 | 限制 |
|---|---|
| 文件格式 | `.md`、`.txt`、文本型 `.pdf`、`.docx` |
| 单次文件数 | 最多 10 个 |
| 单文件大小 | 最大 10 MB |
| 单文件提取文本 | 最多 1,000,000 字符 |
| 单次上传提取文本 | 合计最多 2,000,000 字符 |
| PDF | 只支持能提取文本的 PDF，不包含 OCR |

一次上传会先完成所有文件的解析、分块和向量化，再统一写入索引；如果其中一个文件失败，本次上传不会写入部分成功结果。同名用户文档会替换旧索引。

原始上传文件只在请求处理期间保存在内存中，不会单独落盘；提取后的文本片段、向量和来源元数据会写入 `data/vectors.json`。页面上的删除按钮会删除该用户文档对应的索引片段。

再次运行 `npm run ingest` 会更新内置种子资料，并在 Embedding 模型一致时保留用户上传的索引。更换 Embedding 模型后，旧向量不能混用，需要重新建立内置索引并重新上传用户资料。

## LLM 配置

在 `.env` 中选择一个 `LLM_PROVIDER`：

| 值 | 厂商 | 必填 Key | 默认模型 |
|---|---|---|---|
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| `qwen` | 阿里云百炼 / Qwen | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `zhipu` | 智谱 GLM | `ZHIPU_API_KEY` | `glm-5.2` |
| `moonshot` | Kimi / Moonshot | `MOONSHOT_API_KEY` | `kimi-k2.6` |
| `doubao` | 火山方舟 / 豆包 | `DOUBAO_API_KEY` | 需填写 `DOUBAO_MODEL` |
| `baidu` | 百度千帆 / ERNIE | `BAIDU_API_KEY` | `ernie-4.5-turbo-32k` |
| `custom` | 其他 OpenAI-compatible 服务 | `LLM_API_KEY` | 需填写 `LLM_MODEL` 和 `LLM_BASE_URL` |

模型 ID、地域地址和账号权限可能变化。如果默认模型不可用，请以厂商控制台为准修改相应的 `*_MODEL` 和 `*_BASE_URL`。

## 数据边界

- 文档解析、分块和 Embedding 全部在本应用的 Node.js 服务端完成。
- Embedding 使用本地缓存的 ONNX 模型，不调用云端 Embedding API。
- 用户文件会上传到本应用服务端进行处理。
- RAG 问答会把用户问题和检索出的 Top-K 文本片段发送给所选 LLM 厂商。
- `.env` 中的 API Key 不会通过状态接口返回前端。
- Demo 没有用户认证、权限隔离和多租户能力，不要直接部署为公网多人服务。

## 常用命令

| 命令 | 作用 |
|---|---|
| `npm run ingest` | 下载/加载 ONNX 模型并更新内置种子索引 |
| `npm run dev` | 同时启动 Express 服务端和 Vite 网页端 |
| `npm run dev:server` | 只启动服务端 |
| `npm run dev:web` | 只启动网页端 |
| `npm run build` | 编译服务端并构建网页，作为发布前检查 |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/status` | ONNX 下载/加载状态、LLM 配置状态、文档和 chunk 数量 |
| `POST` | `/api/search` | 混合检索诊断接口 |
| `POST` | `/api/ask` | 会话式 RAG Agent：支持有限 `history`、有界的知识库再次检索和 LLM 生成 |
| `POST` | `/api/documents` | `multipart/form-data` 上传文档，字段名为 `files` |
| `DELETE` | `/api/documents` | 删除用户资料，请求体为 `{ "source": "user:文件名" }` |

`POST /api/ask` 的 `history` 是可选数组，只允许 `user` / `assistant` 两种角色。服务端限制为最多 8 条消息、单条最多 4,000 字符、合计最多 16,000 字符。例如：

```json
{
  "question": "那怎么修复这个问题？",
  "history": [
    { "role": "user", "content": "第一章有什么问题？" },
    { "role": "assistant", "content": "第一章主要存在前置知识过多的问题。" }
  ],
  "topK": 5,
  "vectorWeight": 0.7,
  "keywordWeight": 0.3
}
```

### Agent Loop 的防御边界

- 用户问题、历史消息、工具查询、Top-K 和上传文件都有数量或长度校验。
- 历史仅用于消解指代，历史里的助手回答不能替代知识库证据。
- 知识库片段按不可信数据处理，其中的提示词或命令不能覆盖系统规则。
- 工具采用白名单，只提供只读知识库搜索；参数经过 JSON、类型、长度和范围校验。
- 重复查询会被拒绝，LLM 步骤、单步工具调用数和总来源数均有硬上限。
- `LLM_TIMEOUT_MS` 限制整次 Agent Loop；浏览器取消请求时，服务端也会中止后续 LLM 调用。
- 不支持工具调用的 OpenAI-compatible 模型接口会自动降级为单次 RAG 生成。

## 常见问题

### 页面一直显示 ONNX loading

第一次需要下载约 135 MB 文件。查看服务端终端中的当前文件和进度；如果长时间没有变化，检查 Hugging Face Hub 的网络连通性。

### 页面显示知识库为空

在项目根目录运行 `npm run ingest`，或从网页上传自己的资料。

### 提示索引模型与当前模型不一致

不同 Embedding 模型的向量不能混用。重新运行 `npm run ingest`，并重新上传用户资料。

### PDF 未提取到文本

该 PDF 很可能是扫描图片或加密文件。本 Demo 没有 OCR，需要先使用其他工具把内容转换成可复制文本的 PDF、Markdown 或 TXT。

### LLM 返回 401、403 或模型不存在

核对当前厂商的 API Key、Base URL、模型 ID、账号权限和地域。`.env.example` 中的值是预设起点，最终以厂商控制台为准。

## 项目结构

```text
packages/server/src/
├── config.ts          # 国内 LLM 厂商预设与环境变量校验
├── embedding.ts       # ONNX 模型下载、缓存、状态与 Embedding
├── chunking.ts        # Markdown 感知分块
├── vectorStore.ts     # JSON 持久化、余弦相似度与 BM25
├── search.ts          # 混合召回、归一化与加权融合
├── documents.ts       # PDF/DOCX/MD/TXT 解析与上传索引
├── seedDocuments.ts   # 内置示例资料
├── ingest.ts          # 内置种子索引脚本
├── rag.ts             # Top-K 上下文组装与 LLM 生成
└── index.ts           # Express API

packages/web/src/App.tsx  # Demo 页面
rag-article-outline.md     # 完整文章大纲与 Demo 规格
```

## 安全提示

- 不要提交 `.env`、模型缓存或 `data/vectors.json`。
- 不要把真实敏感资料上传到不受信任的部署实例。
- 调用 RAG 问答前，应确认所选 LLM 厂商的数据处理政策符合使用要求。
- 生产环境还需要认证、权限过滤、审计、限流、恶意文件检测和提示注入防护。
