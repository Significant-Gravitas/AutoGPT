/**
 * Splits streamed prose into the pieces it is revealed in: a word with the
 * whitespace after it, or, in scripts written without spaces (Chinese,
 * Japanese, Thai…), a dictionary word. Splitting those on whitespace alone
 * would hold back or reveal a whole paragraph as one piece.
 *
 * A piece never ends inside a surrogate pair or a combining sequence:
 * whitespace is only cut where no mark follows it, and the word segmenter
 * keeps grapheme clusters whole.
 */

// Leading whitespace, a run of non-whitespace, then the whitespace after it
// unless a combining mark or joiner follows (a space can carry a mark).
const RUN = /\s*\S+(?:\s+(?![\p{M}‍]))?/uy;
const UNSPACED_SCRIPT =
  /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]/u;
// Without Intl.Segmenter: one grapheme-ish piece per base character, with its
// marks and any zero-width-joined characters.
const GRAPHEME_FALLBACK = /\P{M}\p{M}*(?:‍\P{M}\p{M}*)*/gu;

let wordSegmenter: Intl.Segmenter | null | undefined;

function getWordSegmenter() {
  if (wordSegmenter === undefined) {
    wordSegmenter =
      typeof Intl !== "undefined" && "Segmenter" in Intl
        ? new Intl.Segmenter(undefined, { granularity: "word" })
        : null;
  }
  return wordSegmenter;
}

function splitUnspaced(run: string): string[] {
  const segmenter = getWordSegmenter();
  if (segmenter) return Array.from(segmenter.segment(run), (s) => s.segment);
  return run.match(GRAPHEME_FALLBACK) ?? [run];
}

/** Pieces of `text` in order; joined, they are `text` again. */
export function splitTextSegments(text: string): string[] {
  const pieces: string[] = [];
  let index = 0;
  while (index < text.length) {
    RUN.lastIndex = index;
    const match = RUN.exec(text);
    if (!match) {
      // Only whitespace is left.
      pieces.push(text.slice(index));
      break;
    }
    pieces.push(...splitRun(match[0]));
    index = RUN.lastIndex;
  }
  return pieces;
}

function splitRun(run: string): string[] {
  if (!UNSPACED_SCRIPT.test(run)) return [run];
  const start = run.search(/\S/);
  const end = run.trimEnd().length;
  const words = splitUnspaced(run.slice(start, end));
  words[0] = run.slice(0, start) + words[0];
  words[words.length - 1] += run.slice(end);
  return words;
}

/**
 * Offsets after each piece of `text` that is safe to reveal now. The last
 * piece is held unless whitespace ends it: it may be a word, or a character
 * whose marks, still arriving, would join it.
 */
export function findSegmentCutPoints(text: string): number[] {
  const pieces = splitTextSegments(text);
  const cuts: number[] = [];
  let offset = 0;
  for (const piece of pieces) {
    offset += piece.length;
    cuts.push(offset);
  }
  // Whitespace alone also waits, for the word it leads into.
  const last = pieces[pieces.length - 1];
  if (last !== undefined && (!/\s$/.test(last) || !/\S/.test(last))) cuts.pop();
  return cuts;
}
