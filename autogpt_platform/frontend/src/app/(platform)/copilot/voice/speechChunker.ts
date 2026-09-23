/**
 * Splits a streaming reply into speakable pieces.
 *
 * Synthesis of the first sentence starts while the rest is still being
 * written, which is the difference between speaking 0.5 s and 16 s after the
 * reply begins.
 */

export const MAX_CHUNK_CHARS = 400;

/**
 * Every chunk is its own synthesis, so prosody restarts at each seam. Only
 * the first one has to be short — that is the one the wait is measured on.
 */
export const LATER_CHUNK_MIN_CHARS = 200;

interface Result {
  chunks: string[];
  /** Text held back until more arrives (or `flush` is passed). */
  rest: string;
}

/**
 * @param flush - the reply is complete, so emit the trailing partial sentence.
 * @param minChars - hold a chunk back until it is at least this long.
 */
export function takeSpeakableChunks(
  buffer: string,
  flush = false,
  minChars = 0,
): Result {
  const chunks: string[] = [];
  let rest = buffer;

  for (;;) {
    const end = findChunkEnd(rest, flush, minChars);
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

function findChunkEnd(
  text: string,
  atEnd: boolean,
  minChars: number,
): number | null {
  for (let i = 0; i < text.length; i++) {
    const longEnough = i + 1 >= minChars;
    if (longEnough && text[i] === "\n") return i + 1;
    if (longEnough && isSentenceEnd(text, i, atEnd)) return i + 1;
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
