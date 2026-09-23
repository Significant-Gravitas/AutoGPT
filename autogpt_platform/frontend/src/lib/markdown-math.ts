// A backtick fence's info string may not contain a backtick; a closer carries none.
const OPENING_FENCE_RE = /^(?:(`{3,})[^`]*|(~{3,}).*)$/;
const CLOSING_FENCE_RE = /^(`{3,}|~{3,})[ \t]*$/;
const LATEX_SYNTAX_RE = /[\\^_{}]/;
const LIST_MARKER_RE = /^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+/;
const HEADING_RE = /^#{1,6}(?:\s|$)/;
const THEMATIC_BREAK_RE = /^([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const SETEXT_UNDERLINE_RE = /^(?:=+|-+)[ \t]*$/;

// With single-dollar math on, remark-math reads "$5 and $10" as one formula. A "$"
// before a digit is therefore currency unless the span up to the next "$" on that
// line carries LaTeX syntax, which keeps "$5x^2$" math; escaping the rest leaves
// them literal and every other delimiter to remark-math.
export function escapeCurrencyAmounts(markdown: string): string {
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
      return escapeCurrencyAmountsInLine(line);
    })
    .join("\n");
}

function escapeCurrencyAmountsInLine(line: string): string {
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
    } else if (char === "$" && isCurrencyAmount(line, index)) {
      result += "\\$";
      index += 1;
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

function isCurrencyAmount(line: string, index: number): boolean {
  if (!/\d/.test(line[index + 1] ?? "")) {
    return false;
  }

  const rest = line.slice(index + 1);
  const close = rest.search(/(?<!\\)\$/);
  return close === -1 || !LATEX_SYNTAX_RE.test(rest.slice(0, close));
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
