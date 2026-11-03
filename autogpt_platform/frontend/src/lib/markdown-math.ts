import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { gfm } from "micromark-extension-gfm";

const LATEX_SYNTAX_RE = /[\\^_{}]/;
const TEX_COMMAND_RE = /\\[A-Za-z]/;

interface MarkdownNode {
  type: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: MarkdownNode[];
}

// With single-dollar math on, remark-math pairs any two dollars, so "$5 and $10"
// or shell text like "$f ($" became formulas. In the prose the parser finds, a
// "$…$" span on one line that reads as math is kept and every other "$" escaped.
export function escapeLiteralDollars(markdown: string): string {
  const tree = fromMarkdown(markdown, {
    extensions: [gfm()],
    mdastExtensions: [gfmFromMarkdown()],
  });
  let result = "";
  let index = 0;

  for (const [start, end] of proseRanges(tree, markdown)) {
    result += markdown.slice(index, start);
    // A formula kept whole may run past the end of the text node it opened in.
    index = Math.max(start, index);

    while (index < end) {
      const char = markdown[index];
      if (char === "\\") {
        result += markdown.slice(index, index + 2);
        index += 2;
      } else if (char === "$" && markdown[index + 1] === "$") {
        result += "$$";
        index += 2;
      } else if (char === "$") {
        const close = closingDollar(markdown, index);
        result += close === null ? "\\$" : markdown.slice(index, close + 1);
        index = close === null ? index + 1 : close + 1;
      } else {
        result += char;
        index += 1;
      }
    }
  }

  return result + markdown.slice(index);
}

// Source ranges of the text nodes, in order. Code and HTML hold none, and an
// autolink's text is its address.
function proseRanges(node: MarkdownNode, markdown: string): [number, number][] {
  const start = node.position?.start.offset;
  const end = node.position?.end.offset;
  if (node.type === "link" && start !== undefined && markdown[start] !== "[") {
    return [];
  }
  if (node.type === "text") {
    return start === undefined || end === undefined ? [] : [[start, end]];
  }
  return (node.children ?? []).flatMap((child) => proseRanges(child, markdown));
}

// Pandoc's delimiter rule (no space just inside either "$", no digit right after
// the closer), plus: a span opening on a digit is a price unless it carries LaTeX
// syntax, a closer that starts the next shell variable ends shell text, and a
// span with unbalanced brackets and no TeX command is shell text such as "$f ($".
function closingDollar(text: string, open: number): number | null {
  const first = text[open + 1] ?? "";
  if (!first || /\s/.test(first)) return null;

  let close = open + 1;
  while (close < text.length && text[close] !== "$" && text[close] !== "\n") {
    close += text[close] === "\\" && text[close + 1] !== "\n" ? 2 : 1;
  }
  if (text[close] !== "$" || text[close + 1] === "$") return null;
  const after = text[close + 1] ?? "";
  if (/\s/.test(text[close - 1]) || /\d/.test(after)) return null;

  const body = text.slice(open + 1, close);
  if (/\d/.test(first) && !LATEX_SYNTAX_RE.test(body)) return null;
  if (startsShellVariable(after, body)) return null;
  return TEX_COMMAND_RE.test(body) || bracketsBalance(body) ? close : null;
}

// "$HOME/bin:$PATH", "${A}:${B}", "$HOME$PATH"; a one-letter body keeps "$n$th".
function startsShellVariable(after: string, body: string): boolean {
  return /[A-Za-z_{(]/.test(after) && body.length > 1;
}

// Openers and closers of either kind count alike, so "[0, 1)" balances.
function bracketsBalance(text: string): boolean {
  let depth = 0;
  for (const char of text) {
    if (char === "(" || char === "[") depth += 1;
    if (char === ")" || char === "]") depth -= 1;
    if (depth < 0) return false;
  }
  return depth === 0;
}
