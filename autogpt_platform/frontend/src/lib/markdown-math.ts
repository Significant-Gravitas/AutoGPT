// A backtick fence's info string may not contain a backtick; a closer carries none.
const OPENING_FENCE_RE = /^(?:(`{3,})[^`]*|(~{3,}).*)$/;
const CLOSING_FENCE_RE = /^(`{3,}|~{3,})[ \t]*$/;
const QUOTE_PREFIX_RE = /^(?: {0,3}>[ \t]?)*/;
const LATEX_SYNTAX_RE = /[\\^_{}]/;
const TEX_COMMAND_RE = /\\[A-Za-z]/;
const SHELL_VARIABLE_RE = /^[A-Za-z_]\w*[^\w\s]$/;
const LIST_MARKER_RE = /^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+/;
const HEADING_RE = /^#{1,6}(?:\s|$)/;
const THEMATIC_BREAK_RE = /^([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const SETEXT_UNDERLINE_RE = /^(?:=+|-+)[ \t]*$/;
// CommonMark HTML blocks: these four run to their closing tag, the rest to a blank line.
const HTML_RAW_RE = /^<(pre|script|style|textarea)(?=[\s>]|$)/i;
const HTML_BLOCK_TAG_RE =
  /^<\/?(?:address|article|aside|blockquote|details|dialog|div|dl|fieldset|figcaption|figure|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|section|summary|table|tbody|td|tfoot|th|thead|tr|ul)(?=[\s/>]|$)/i;
const HTML_TAG_LINE_RE =
  /^(?:<[A-Za-z][\w-]*(?:\s[^<>]*)?\/?>|<\/[A-Za-z][\w-]*\s*>)\s*$/;
const AUTOLINK_RE = /<[A-Za-z][\w+.-]{1,31}:[^\s<>]*>/y;
const BARE_URL_RE = /(?:https?:\/\/|www\.)[^\s<]+/iy;

type HTMLBlockEnd = RegExp | "blank";

interface CodeSpan {
  run: string | null;
}

// With single-dollar math on, remark-math pairs any two dollars, so "$5 and $10"
// or shell text like "$f ($" became formulas. Only a "$…$" span on one line that
// reads as math is kept for it; every other lone "$" in prose is escaped.
export function escapeLiteralDollars(markdown: string): string {
  const lines = markdown.split("\n");
  let openFence: { fence: string; depth: number } | null = null;
  let codeIndent: number | null = null;
  let htmlEnd: HTMLBlockEnd | null = null;
  // Content column of each open list item, innermost last.
  const listIndents: number[] = [];
  let inParagraph = false;
  const codeSpan: CodeSpan = { run: null };
  const listIndent = () => listIndents.at(-1) ?? 0;

  function escapeBlockLine(line: string, depth: number, at: number): string {
    const indent = indentWidth(line);
    const content = line.trimStart();

    if (openFence) {
      if (depth > openFence.depth) return line;
      if (depth === openFence.depth) {
        const fence =
          indent < listIndent() + 4
            ? CLOSING_FENCE_RE.exec(content)?.[1]
            : undefined;
        if (
          fence &&
          fence[0] === openFence.fence[0] &&
          fence.length >= openFence.fence.length
        ) {
          openFence = null;
        }
        return line;
      }
      // The blockquote holding the fence has ended, and the fence with it.
      openFence = null;
    }

    if (htmlEnd instanceof RegExp) {
      if (htmlEnd.test(line)) htmlEnd = null;
      return line;
    }

    if (!line.trim()) {
      inParagraph = false;
      htmlEnd = null;
      codeSpan.run = null;
      return line;
    }

    if (htmlEnd) return line;

    if (codeIndent !== null && indent >= codeIndent) return line;
    codeIndent = null;

    const marker = LIST_MARKER_RE.exec(line);
    const startsItem = marker !== null && indent < listIndent() + 4;
    // A lazy paragraph line stays in its list item; anything else closes the
    // items it is not indented into.
    if (startsItem || !inParagraph) {
      while (listIndents.length && listIndent() > indent) listIndents.pop();
    }

    // Block markers may be indented up to three columns past the list item's content.
    const startsBlock = indent < listIndent() + 4;
    const opening = startsBlock ? OPENING_FENCE_RE.exec(content) : null;
    const fence = opening ? (opening[1] ?? opening[2]) : undefined;

    if (fence) {
      openFence = { fence, depth };
      inParagraph = false;
      return line;
    }

    if (
      startsBlock &&
      (THEMATIC_BREAK_RE.test(content) ||
        (inParagraph && SETEXT_UNDERLINE_RE.test(content)))
    ) {
      inParagraph = false;
      return line;
    }

    if (startsItem) listIndents.push(columns(marker[0]));

    // Indented code cannot interrupt a paragraph, and CommonMark renders a
    // backslash escape inside it, or in an HTML block, literally.
    if (!inParagraph && indent >= listIndent() + 4) {
      codeIndent = listIndent() + 4;
      return line;
    }

    const html = startsBlock ? htmlBlockEnd(content, inParagraph) : null;
    if (html) {
      htmlEnd = html instanceof RegExp && html.test(content) ? null : html;
      inParagraph = false;
      return line;
    }

    inParagraph = !(startsBlock && HEADING_RE.test(content));
    return escapeLiteralDollarsInLine(line, codeSpan, () =>
      paragraphAfter(lines, at),
    );
  }

  return lines
    .map((raw, at) => {
      const quote = QUOTE_PREFIX_RE.exec(raw)?.[0] ?? "";
      const depth = quote.split(">").length - 1;
      return quote + escapeBlockLine(raw.slice(quote.length), depth, at);
    })
    .join("\n");
}

function escapeLiteralDollarsInLine(
  line: string,
  codeSpan: CodeSpan,
  rest: () => string,
): string {
  let result = "";
  let index = 0;

  if (codeSpan.run) {
    const close = line.indexOf(codeSpan.run);
    if (close === -1) return line;
    index = close + codeSpan.run.length;
    result = line.slice(0, index);
    codeSpan.run = null;
  }

  while (index < line.length) {
    const char = line[index];
    const linkEnd = endOfLink(line, index);

    if (linkEnd !== null) {
      result += line.slice(index, linkEnd);
      index = linkEnd;
    } else if (char === "\\") {
      result += line.slice(index, index + 2);
      index += 2;
    } else if (char === "`") {
      const run = backtickRun(line, index);
      const close = line.indexOf(run, index + run.length);
      // A code span may close on a later line of its paragraph.
      if (close === -1 && rest().includes(run)) {
        codeSpan.run = run;
        return result + line.slice(index);
      }
      const end = close === -1 ? index + run.length : close + run.length;
      result += line.slice(index, end);
      index = end;
    } else if (char === "$" && line[index + 1] === "$") {
      result += "$$";
      index += 2;
    } else if (char === "$") {
      const close = closingDollar(line, index);
      result += close === null ? "\\$" : line.slice(index, close + 1);
      index = close === null ? index + 1 : close + 1;
    } else {
      result += char;
      index += 1;
    }
  }

  return result;
}

// An autolink or a bare URL keeps a backslash in its address, so it is copied as is.
function endOfLink(line: string, start: number): number | null {
  const pattern =
    line[start] === "<"
      ? AUTOLINK_RE
      : /[hw]/i.test(line[start]) && !/\w/.test(line[start - 1] ?? "")
        ? BARE_URL_RE
        : null;
  if (!pattern) return null;
  pattern.lastIndex = start;
  const match = pattern.exec(line);
  return match ? start + match[0].length : null;
}

function backtickRun(line: string, start: number): string {
  let end = start;
  while (line[end] === "`") end += 1;
  return line.slice(start, end);
}

// The rest of the paragraph after line `at`, up to its first blank line.
function paragraphAfter(lines: string[], at: number): string {
  const rest = lines.slice(at + 1);
  const blank = rest.findIndex((line) => !line.trim());
  return rest.slice(0, blank === -1 ? undefined : blank).join("\n");
}

function htmlBlockEnd(
  content: string,
  inParagraph: boolean,
): HTMLBlockEnd | null {
  const raw = HTML_RAW_RE.exec(content);
  if (raw) return new RegExp(`</${raw[1]}>`, "i");
  if (content.startsWith("<!--")) return /-->/;
  if (HTML_BLOCK_TAG_RE.test(content)) return "blank";
  // Any other tag starts a block only on a line of its own, outside a paragraph.
  return !inParagraph && HTML_TAG_LINE_RE.test(content) ? "blank" : null;
}

// Pandoc's delimiter rule (no space just inside either "$", no digit right after
// the closer), plus: a span opening on a digit is a price unless it carries LaTeX
// syntax, "$HOME/$" joins two shell variables, and a span with unbalanced
// brackets and no TeX command is shell text such as "$f ($".
function closingDollar(line: string, open: number): number | null {
  const first = line[open + 1] ?? "";
  if (!first || /\s/.test(first)) return null;

  let close = open + 1;
  while (close < line.length && line[close] !== "$") {
    close += line[close] === "\\" ? 2 : 1;
  }
  if (close >= line.length || line[close + 1] === "$") return null;
  const after = line[close + 1] ?? "";
  if (/\s/.test(line[close - 1]) || /\d/.test(after)) return null;

  const body = line.slice(open + 1, close);
  if (/\d/.test(first) && !LATEX_SYNTAX_RE.test(body)) return null;
  if (SHELL_VARIABLE_RE.test(body) && /[A-Za-z_{(]/.test(after)) return null;
  return TEX_COMMAND_RE.test(body) || bracketsBalance(body) ? close : null;
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

function indentWidth(line: string): number {
  return columns(line.slice(0, line.length - line.trimStart().length));
}

// A tab advances to the next multiple of four columns.
function columns(text: string): number {
  let width = 0;
  for (const char of text) width += char === "\t" ? 4 - (width % 4) : 1;
  return width;
}
