// A backtick fence's info string may not contain a backtick; a closer carries none.
const OPENING_FENCE_RE = /^(?:(`{3,})[^`]*|(~{3,}).*)$/;
const CLOSING_FENCE_RE = /^(`{3,}|~{3,})[ \t]*$/;
const LATEX_SYNTAX_RE = /[\\^_{}]/;
const LIST_MARKER_RE = /^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+/;
const HEADING_RE = /^#{1,6}(?:\s|$)/;
const THEMATIC_BREAK_RE = /^([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const SETEXT_UNDERLINE_RE = /^(?:=+|-+)[ \t]*$/;

// With single-dollar math on, remark-math pairs any two dollars, so "$5 and $10"
// or shell text like "$p ($(ps -p $p))" became formulas. Only a "$…$" span on one
// line that reads as math is kept for it; every other lone "$" is escaped.
export function escapeLiteralDollars(markdown: string): string {
  let openFence: string | null = null;
  let codeIndent: number | null = null;
  // Content column of each open list item, innermost last.
  const listIndents: number[] = [];
  let inParagraph = false;
  const listIndent = () => listIndents.at(-1) ?? 0;

  return markdown
    .split("\n")
    .map((line) => {
      const indent = indentWidth(line);
      const content = line.trimStart();

      if (openFence) {
        const fence =
          indent < listIndent() + 4
            ? CLOSING_FENCE_RE.exec(content)?.[1]
            : undefined;
        if (
          fence &&
          fence[0] === openFence[0] &&
          fence.length >= openFence.length
        ) {
          openFence = null;
        }
        return line;
      }

      if (!line.trim()) {
        inParagraph = false;
        return line;
      }

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
        openFence = fence;
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
      // backslash escape inside it literally.
      if (!inParagraph && indent >= listIndent() + 4) {
        codeIndent = listIndent() + 4;
        return line;
      }

      inParagraph = !(startsBlock && HEADING_RE.test(content));
      return escapeLiteralDollarsInLine(line);
    })
    .join("\n");
}

function escapeLiteralDollarsInLine(line: string): string {
  let result = "";
  let index = 0;

  while (index < line.length) {
    const char = line[index];

    if (char === "\\") {
      result += line.slice(index, index + 2);
      index += 2;
    } else if (char === "`") {
      const end = endOfCodeSpan(line, index);
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

function endOfCodeSpan(line: string, start: number): number {
  let runEnd = start;
  while (line[runEnd] === "`") {
    runEnd += 1;
  }

  const close = line.indexOf(line.slice(start, runEnd), runEnd);
  return close === -1 ? runEnd : close + runEnd - start;
}

// Pandoc's delimiter rule (no space just inside either "$", no digit right after
// the closer), plus: a span opening on a digit is a price unless it carries LaTeX
// syntax, and a span with unbalanced brackets is shell text such as "$p ($".
function closingDollar(line: string, open: number): number | null {
  const first = line[open + 1] ?? "";
  if (!first || /\s/.test(first)) return null;

  let close = open + 1;
  while (close < line.length && line[close] !== "$") {
    close += line[close] === "\\" ? 2 : 1;
  }
  if (close >= line.length || line[close + 1] === "$") return null;
  if (/\s/.test(line[close - 1]) || /\d/.test(line[close + 1] ?? "")) {
    return null;
  }

  const body = line.slice(open + 1, close);
  if (/\d/.test(first) && !LATEX_SYNTAX_RE.test(body)) return null;
  return bracketsBalance(body) ? close : null;
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
