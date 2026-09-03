/**
 * Splits a streaming reply into speakable pieces.
 *
 * Synthesis of the first sentence starts while the rest is still being
 * written, which is the difference between speaking 0.5 s and 16 s after the
 * reply begins.
 */

export const MAX_CHUNK_CHARS = 200;

interface Result {
  chunks: string[];
  /** Text held back until more arrives (or `flush` is passed). */
  rest: string;
}

/**
 * @param flush - the reply is complete, so emit the trailing partial sentence.
 */
export function takeSpeakableChunks(buffer: string, flush = false): Result {
  const chunks: string[] = [];
  let rest = buffer;

  for (;;) {
    const end = findChunkEnd(rest, flush);
    if (end === null) break;
    const chunk = rest.slice(0, end).trim();
    rest = rest.slice(end).trimStart();
    if (chunk) chunks.push(chunk);
  }

  if (flush) {
    const tail = rest.trim();
    if (tail) chunks.push(tail);
    rest = "";
  }

  return { chunks, rest };
}

function findChunkEnd(text: string, atEnd: boolean): number | null {
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "\n") return i + 1;
    if (isSentenceEnd(text, i, atEnd)) return i + 1;
    // Nothing has ended a sentence in a whole chunk's worth of text — a long
    // list or a code-free wall of prose. Break on the last word boundary so
    // the split is not mid-word.
    if (i + 1 >= MAX_CHUNK_CHARS) {
      const space = text.lastIndexOf(" ", MAX_CHUNK_CHARS);
      return space > 0 ? space + 1 : MAX_CHUNK_CHARS;
    }
  }
  return null;
}

function isSentenceEnd(text: string, i: number, atEnd: boolean): boolean {
  if (text[i] !== "." && text[i] !== "!" && text[i] !== "?") return false;
  const next = text[i + 1];
  // Mid-stream, a terminator at the very end of the buffer is unproven: the
  // next delta may turn "3." into "3.5".
  if (next === undefined) return atEnd;
  return /\s/.test(next);
}
