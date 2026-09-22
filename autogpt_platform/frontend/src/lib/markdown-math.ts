const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
const LATEX_SYNTAX_RE = /[\\^_{}]/;
const LIST_MARKER_RE = /^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+/;
const HEADING_RE = /^ {0,3}#{1,6}(?:\s|$)/;

// With single-dollar math on, remark-math reads "$5 and $10" as one formula. A "$"
// before a digit is therefore currency unless the span up to the next "$" on that
// line carries LaTeX syntax, which keeps "$5x^2$" math; escaping the rest leaves
// them literal and every other delimiter to remark-math.
export function escapeCurrencyAmounts(markdown: string): string {
  let openFence: string | null = null;
  let codeIndent: number | null = null;
  let listIndent = 0;
  let inParagraph = false;

  return markdown
    .split("\n")
    .map((line) => {
      const fence = FENCE_RE.exec(line)?.[1];

      if (openFence) {
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

      const indent = indentWidth(line);
      if (codeIndent !== null && indent >= codeIndent) return line;
      codeIndent = null;

      if (fence) {
        openFence = fence;
        inParagraph = false;
        return line;
      }

      const marker = LIST_MARKER_RE.exec(line);
      if (marker && indent < listIndent + 4) {
        listIndent = marker[0].length;
      } else if (indent < listIndent && !inParagraph) {
        listIndent = 0;
      }

      // Indented code cannot interrupt a paragraph, and CommonMark renders a
      // backslash escape inside it literally.
      if (!marker && !inParagraph && indent >= listIndent + 4) {
        codeIndent = listIndent + 4;
        return line;
      }

      inParagraph = !HEADING_RE.test(line);
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
  let width = 0;
  for (const char of line) {
    if (char === " ") width += 1;
    else if (char === "\t") width += 4 - (width % 4);
    else break;
  }
  return width;
}
