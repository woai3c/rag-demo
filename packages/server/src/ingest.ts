import './config.js';
import { chunkMarkdown } from './chunking.js';
import { embedPassage, initializeEmbedding, getEmbeddingInfo, EMBEDDING_INDEX_ID } from './embedding.js';
import { addChunks, loadStore, removeBuiltInChunks, saveStore, type StoredChunk } from './vectorStore.js';
import { SEED_DOCUMENTS } from './seedDocuments.js';

async function main() {
  console.log('=== RAG Demo: 文档索引 ===\n');

  await initializeEmbedding();
  const { provider, model, dimensions } = getEmbeddingInfo();
  console.log(`Embedding 方案: ${provider} / ${model} (${dimensions} 维)\n`);

  console.log(`准备索引 ${SEED_DOCUMENTS.length} 份内置虚构资料:\n`);
  for (const document of SEED_DOCUMENTS) {
    console.log(`  - ${document.source}`);
  }
  console.log();

  loadStore(EMBEDDING_INDEX_ID);
  const removed = removeBuiltInChunks();
  console.log(`替换 ${removed} 个旧的内置 chunks；保留用户上传资料`);

  let totalChunks = 0;
  for (const document of SEED_DOCUMENTS) {
    const chunks = chunkMarkdown(document.content, document.source);
    console.log(`[${document.source}] 切分为 ${chunks.length} 个 chunks`);

    const storedChunks: StoredChunk[] = [];
    for (const chunk of chunks) {
      const vector = await embedPassage(chunk.content);
      storedChunks.push({ ...chunk, vector });
    }

    addChunks(storedChunks);
    totalChunks += chunks.length;
  }

  saveStore(EMBEDDING_INDEX_ID);
  console.log(`\n完成！共索引 ${totalChunks} 个 chunks`);
}

main().catch(error => {
  console.error('[Ingest Error]', error);
  process.exitCode = 1;
});
