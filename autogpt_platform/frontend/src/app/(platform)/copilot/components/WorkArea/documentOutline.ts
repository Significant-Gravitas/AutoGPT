const HEADING = /^(#{1,3})\s+(.+?)\s*#*$/;
const FENCE = /^\s*(```|~~~)/;
const MARKUP = /[*_`]/g;

export interface OutlineEntry {
  text: string;
  level: number;
}

/** The headings of a markdown document, down to the third level, skipping
 *  anything inside a code block. */
export function outlineOf(markdown: string): OutlineEntry[] {
  const entries: OutlineEntry[] = [];
  let isInCode = false;
  for (const line of markdown.split("\n")) {
    if (FENCE.test(line)) {
      isInCode = !isInCode;
      continue;
    }
    if (isInCode) continue;
    const match = HEADING.exec(line);
    if (!match) continue;
    const text = match[2].replace(MARKUP, "").trim();
    if (text) entries.push({ text, level: match[1].length });
  }
  return entries;
}

/** The heading an outline entry points at, inside a rendered document. */
export function headingFor(
  root: HTMLElement | null,
  entry: OutlineEntry,
  position: number,
) {
  const headings = Array.from(root?.querySelectorAll("h1, h2, h3") ?? []);
  return (
    headings.find((heading) => heading.textContent?.trim() === entry.text) ??
    headings[position] ??
    null
  );
}
