const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
// CommonMark: a closing fence is only the marker, then whitespace.
const CLOSING_FENCE_RE = /^\s{0,3}(`{3,}|~{3,})\s*$/;
// `[label]: url`, which a link anywhere in the document may point to.
const REFERENCE_RE = /^\s{0,3}\[[^\]]+\]:\s*\S/m;
const LIST_ITEM_RE = /^\s*([-*+]|\d+[.)])\s/;

/**
 * Splits markdown at the blank lines between top-level blocks, so a streaming
 * reply re-parses only its last block. Code fences, lists and indented
 * continuations stay whole, since splitting them would change how they render.
 */
export function splitMarkdownBlocks(text: string): string[] {
  // Each block is parsed on its own, so a link would lose its reference.
  if (REFERENCE_RE.test(text)) return [text];
  const blocks: string[] = [];
  let current: string[] = [];
  let fence: string | null = null;
  let afterBlank = false;

  for (const line of text.split("\n")) {
    const opener = FENCE_RE.exec(line)?.[1];
    if (fence === null && afterBlank && current.length > 0 && startsBlock(line)) {
      blocks.push(current.join("\n"));
      current = [];
    }
    current.push(line);
    if (fence !== null) {
      const closer = CLOSING_FENCE_RE.exec(line)?.[1];
      if (closer && closer[0] === fence[0] && closer.length >= fence.length) fence = null;
    } else if (opener) {
      fence = opener;
    }
    afterBlank = fence === null && line.trim() === "";
  }
  if (current.length > 0) blocks.push(current.join("\n"));
  return blocks;
}

// A line that opens a new block: not indented (a continuation) and not a list
// item (which may belong to the list above).
function startsBlock(line: string) {
  if (line.trim() === "") return false;
  if (/^\s/.test(line)) return false;
  return !LIST_ITEM_RE.test(line);
}
