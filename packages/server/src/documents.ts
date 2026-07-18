import { extname, basename } from 'path';
import mammoth from 'mammoth';
import pdfParse from 'pdf-parse';
import { chunkMarkdown } from './chunking.js';
import { embedPassages, EMBEDDING_INDEX_ID } from './embedding.js';
import { addChunks, removeBySource, saveStore, type StoredChunk } from './vectorStore.js';

const SUPPORTED_EXTENSIONS = new Set(['.md', '.txt', '.pdf', '.docx']);
export const MAX_EXTRACTED_CHARACTERS_PER_FILE = 1_000_000;
export const MAX_EXTRACTED_CHARACTERS_PER_UPLOAD = 2_000_000;

export interface IngestedDocument {
  source: string;
  characters: number;
  chunks: number;
}

function decodeFileName(name: string): string {
  const decoded = Buffer.from(name, 'latin1').toString('utf8');
  return decoded.includes('\uFFFD') ? name : decoded;
}

function safeFileName(originalName: string): string {
  return basename(decodeFileName(originalName))
    .replace(/[\u0000-\u001f<>:"/\\|?*]/g, '_')
    .slice(0, 180);
}

export function isSupportedDocument(fileName: string): boolean {
  return SUPPORTED_EXTENSIONS.has(extname(fileName).toLowerCase());
}

async function extractText(file: Express.Multer.File): Promise<string> {
  const extension = extname(file.originalname).toLowerCase();

  if (extension === '.md' || extension === '.txt') {
    return file.buffer.toString('utf8');
  }
  if (extension === '.pdf') {
    const result = await pdfParse(file.buffer);
    return result.text;
  }
  if (extension === '.docx') {
    const result = await mammoth.extractRawText({ buffer: file.buffer });
    return result.value;
  }

  throw new Error(`不支持的文件类型：${extension || '未知'}`);
}

interface PreparedDocument {
  document: IngestedDocument;
  chunks: StoredChunk[];
}

async function prepareUploadedFile(file: Express.Multer.File): Promise<PreparedDocument> {
  if (!isSupportedDocument(file.originalname)) {
    throw new Error(`不支持 ${file.originalname}；仅支持 .md、.txt、文本型 .pdf 和 .docx`);
  }

  const text = (await extractText(file)).replace(/\u0000/g, '').trim();
  if (!text) {
    throw new Error(`${file.originalname} 未提取到文本；扫描型 PDF 暂不支持 OCR`);
  }
  if (text.length > MAX_EXTRACTED_CHARACTERS_PER_FILE) {
    throw new Error(`${file.originalname} 提取出 ${text.length} 个字符，超过单文件 ${MAX_EXTRACTED_CHARACTERS_PER_FILE} 字符限制`);
  }

  const source = `user:${safeFileName(file.originalname)}`;
  const chunks = chunkMarkdown(text, source);
  if (chunks.length === 0) {
    throw new Error(`${file.originalname} 没有可索引的内容`);
  }

  const vectors = await embedPassages(chunks.map(chunk => chunk.content));
  const storedChunks: StoredChunk[] = chunks.map((chunk, index) => ({
    ...chunk,
    vector: vectors[index],
  }));

  return {
    document: {
      source,
      characters: text.length,
      chunks: chunks.length,
    },
    chunks: storedChunks,
  };
}

export async function ingestUploadedFiles(files: Express.Multer.File[]): Promise<IngestedDocument[]> {
  const prepared: PreparedDocument[] = [];
  const sources = new Set<string>();
  let totalCharacters = 0;

  for (const file of files) {
    const item = await prepareUploadedFile(file);
    if (sources.has(item.document.source)) {
      throw new Error(`一次上传中存在重名文档：${item.document.source.replace(/^user:/, '')}`);
    }
    sources.add(item.document.source);
    totalCharacters += item.document.characters;
    if (totalCharacters > MAX_EXTRACTED_CHARACTERS_PER_UPLOAD) {
      throw new Error(`本次上传共提取 ${totalCharacters} 个字符，超过 ${MAX_EXTRACTED_CHARACTERS_PER_UPLOAD} 字符限制`);
    }
    prepared.push(item);
  }

  for (const item of prepared) {
    removeBySource(item.document.source);
    addChunks(item.chunks);
  }
  saveStore(EMBEDDING_INDEX_ID);
  return prepared.map(item => item.document);
}
