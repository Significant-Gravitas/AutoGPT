/**
 * A deliberately small Markdown subset for assistant replies, rendered with
 * native text. Anything outside the subset stays readable as plain text.
 * Links are only ever made tappable for http(s) targets.
 */
export type Inline =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "strong" | "em" | "strike"; children: Inline[] }
  | { kind: "link"; href: string | null; children: Inline[] };

export type Block =
  | { kind: "paragraph"; inline: Inline[] }
  | { kind: "heading"; level: number; inline: Inline[] }
  | { kind: "code"; language: string | null; text: string }
  | { kind: "list"; ordered: boolean; start: number; items: Inline[][] }
  | { kind: "quote"; inline: Inline[] }
  | { kind: "preformatted"; text: string }
  | { kind: "rule" };

const FENCE = /^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const LIST_ITEM = /^(\s*)([-*+]|(\d{1,9})[.)])\s+(.*)$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

export function parseMarkdown(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const fence = line.match(FENCE);
    if (fence) {
      const body: string[] = [];
      index += 1;
      // An unterminated fence (a reply still streaming) runs to the end.
      while (
        index < lines.length &&
        !lines[index].trim().startsWith(fence[1])
      ) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1;
      blocks.push({
        kind: "code",
        language: fence[2] || null,
        text: body.join("\n"),
      });
      continue;
    }
    const heading = line.match(HEADING);
    if (heading) {
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        inline: parseInline(heading[2]),
      });
      index += 1;
      continue;
    }
    if (RULE.test(line)) {
      blocks.push({ kind: "rule" });
      index += 1;
      continue;
    }
    if (
      line.includes("|") &&
      index + 1 < lines.length &&
      TABLE_DIVIDER.test(lines[index + 1])
    ) {
      const rows: string[] = [];
      while (index < lines.length && lines[index].includes("|")) {
        rows.push(lines[index].trim());
        index += 1;
      }
      blocks.push({ kind: "preformatted", text: rows.join("\n") });
      continue;
    }
    const item = line.match(LIST_ITEM);
    if (item) {
      index = parseList(lines, index, blocks);
      continue;
    }
    if (QUOTE.test(line)) {
      const quoted: string[] = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        quoted.push((lines[index].match(QUOTE) as RegExpMatchArray)[1]);
        index += 1;
      }
      blocks.push({ kind: "quote", inline: parseInline(quoted.join("\n")) });
      continue;
    }
    // The first line is always consumed, so the loop can't stall.
    const paragraph: string[] = [line.trim()];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !startsBlock(lines, index)
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push({
      kind: "paragraph",
      inline: parseInline(paragraph.join("\n")),
    });
  }
  return blocks;
}

function parseList(lines: string[], start: number, blocks: Block[]): number {
  const first = lines[start].match(LIST_ITEM) as RegExpMatchArray;
  const ordered = first[3] !== undefined;
  const items: string[] = [];
  let index = start;
  while (index < lines.length) {
    const line = lines[index];
    const item = line.match(LIST_ITEM);
    if (item && (item[3] !== undefined) === ordered) {
      items.push(item[4]);
    } else if (item || !line.trim() || startsBlock(lines, index)) {
      // A blank line followed by another item continues the same list.
      const next = lines[index + 1]?.match(LIST_ITEM);
      if (!line.trim() && next && (next[3] !== undefined) === ordered) {
        index += 1;
        continue;
      }
      break;
    } else {
      items[items.length - 1] += `\n${line.trim()}`;
    }
    index += 1;
  }
  blocks.push({
    kind: "list",
    ordered,
    start: ordered ? Number(first[3]) : 1,
    items: items.map(parseInline),
  });
  return index;
}

function startsBlock(lines: string[], index: number): boolean {
  const line = lines[index];
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    LIST_ITEM.test(line) ||
    QUOTE.test(line) ||
    (line.includes("|") &&
      index + 1 < lines.length &&
      TABLE_DIVIDER.test(lines[index + 1]))
  );
}

const INLINE =
  /(`[^`\n]+`)|(\[[^\]\n]+\]\([^)\s]+(?:\s+"[^"]*")?\))|(\*\*[^*\n]+?\*\*)|(__[^_\n]+?__)|(~~[^~\n]+?~~)|(\*[^*\s][^*\n]*?\*)|((?<![\w])_[^_\s][^_\n]*?_(?![\w]))|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g;

export function parseInline(text: string): Inline[] {
  const result: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const start = match.index ?? 0;
    if (start > last)
      result.push({ kind: "text", text: text.slice(last, start) });
    result.push(inlineToken(match));
    last = start + match[0].length;
  }
  if (last < text.length) result.push({ kind: "text", text: text.slice(last) });
  return result;
}

function inlineToken(match: RegExpMatchArray): Inline {
  const [
    token,
    code,
    link,
    strong,
    strongUnderscore,
    strike,
    em,
    emUnderscore,
    url,
  ] = match;
  if (code) return { kind: "code", text: code.slice(1, -1) };
  if (link) {
    const parts = link.match(/^\[([^\]]+)\]\(([^)\s]+)/) as RegExpMatchArray;
    return {
      kind: "link",
      href: safeHref(parts[2]),
      children: parseInline(parts[1]),
    };
  }
  if (strong || strongUnderscore)
    return { kind: "strong", children: parseInline(token.slice(2, -2)) };
  if (strike)
    return { kind: "strike", children: parseInline(token.slice(2, -2)) };
  if (em || emUnderscore)
    return { kind: "em", children: parseInline(token.slice(1, -1)) };
  if (url)
    return {
      kind: "link",
      href: safeHref(url),
      children: [{ kind: "text", text: url }],
    };
  return { kind: "text", text: token };
}

/** Only absolute http(s) links open; workspace, script and app links stay inert. */
function safeHref(raw: string): string | null {
  try {
    const url = new URL(raw);
    return url.protocol === "https:" || url.protocol === "http:"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}
