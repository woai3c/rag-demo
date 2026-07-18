export interface Chunk {
  id: string;
  content: string;
  source: string;
  heading?: string;
  startOffset: number;
}

export interface ChunkingOptions {
  chunkSize?: number;
  chunkOverlap?: number;
}

const DEFAULT_CHUNK_SIZE = 400;
const DEFAULT_CHUNK_OVERLAP = 80;

function hashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

/**
 * Markdown-aware chunking: splits by headings first, then by size.
 * Preserves heading context for each chunk.
 */
export function chunkMarkdown(
  text: string,
  source: string,
  options: ChunkingOptions = {}
): Chunk[] {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const chunkOverlap = options.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;
  if (!Number.isFinite(chunkSize) || chunkSize <= 0) {
    throw new Error('chunkSize 必须是大于 0 的数字');
  }
  if (!Number.isFinite(chunkOverlap) || chunkOverlap < 0 || chunkOverlap >= chunkSize) {
    throw new Error('chunkOverlap 必须大于等于 0 且小于 chunkSize');
  }

  const sections = splitByHeadings(text);
  const chunks: Chunk[] = [];

  for (const section of sections) {
    const sectionChunks = splitBySize(
      section.content,
      source,
      section.heading,
      section.offset,
      chunkSize,
      chunkOverlap
    );
    chunks.push(...sectionChunks);
  }

  return chunks;
}

interface Section {
  heading?: string;
  content: string;
  offset: number;
}

function splitByHeadings(text: string): Section[] {
  const lines = text.split('\n');
  const sections: Section[] = [];
  let currentHeading: string | undefined;
  let currentContent: string[] = [];
  let currentOffset = 0;
  let charOffset = 0;

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s+(.+)/);

    if (headingMatch) {
      if (currentContent.length > 0) {
        const content = currentContent.join('\n').trim();
        if (content) {
          sections.push({
            heading: currentHeading,
            content,
            offset: currentOffset,
          });
        }
      }
      currentHeading = headingMatch[1];
      currentContent = [];
      currentOffset = charOffset + line.length + 1;
    } else {
      currentContent.push(line);
    }

    charOffset += line.length + 1;
  }

  if (currentContent.length > 0) {
    const content = currentContent.join('\n').trim();
    if (content) {
      sections.push({
        heading: currentHeading,
        content,
        offset: currentOffset,
      });
    }
  }

  return sections;
}

function splitBySize(
  text: string,
  source: string,
  heading: string | undefined,
  baseOffset: number,
  chunkSize: number,
  chunkOverlap: number
): Chunk[] {
  const tokens = estimateTokens(text);
  if (tokens <= chunkSize) {
    return [{
      id: `${source}-${baseOffset}-${hashContent(text)}`,
      content: text,
      source,
      heading,
      startOffset: baseOffset,
    }];
  }

  const chunks: Chunk[] = [];
  const charChunkSize = Math.floor(chunkSize * 3.5);
  const charOverlap = Math.floor(chunkOverlap * 3.5);
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + charChunkSize, text.length);

    if (end < text.length) {
      const paragraphBreak = text.lastIndexOf('\n\n', end);
      if (paragraphBreak > start + charChunkSize * 0.5) {
        end = paragraphBreak;
      } else {
        const lineBreak = text.lastIndexOf('\n', end);
        if (lineBreak > start + charChunkSize * 0.5) {
          end = lineBreak;
        }
      }
    }

    const content = text.slice(start, end).trim();
    if (content) {
      chunks.push({
        id: `${source}-${baseOffset + start}-${hashContent(content)}`,
        content,
        source,
        heading,
        startOffset: baseOffset + start,
      });
    }

    start = end - charOverlap;
    if (start >= text.length) break;
  }

  return chunks;
}
