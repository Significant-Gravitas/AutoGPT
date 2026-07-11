import { CHANGELOG_BASE_URL, CHANGELOG_PROXY_URL } from "./changelog-constants";

export interface ChangelogEntry {
  slug: string;
  dateRange: string;
  highlights: string;
  url: string;
  mdUrl: string;
}

export function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
  // Each row links to a release's docs page — absolute or relative, with or
  // without a `.md` suffix, e.g.
  //   | [May 7 – June 10](/docs/platform/changelog/changelog/may-7-june-10-2026.md) | ... |
  const rowPattern =
    /\|\s*\[([^\]]+)\]\([^)]*\/changelog\/changelog\/([a-z0-9-]+)(?:\.md)?\)\s*\|\s*([^|]+)\|/g;

  let match;
  while ((match = rowPattern.exec(md)) !== null) {
    const [, dateRange, slug, highlights] = match;
    entries.push({
      slug,
      dateRange: dateRange.trim(),
      highlights: highlights.trim(),
      url: `${CHANGELOG_BASE_URL}/${slug}`,
      mdUrl: `${CHANGELOG_PROXY_URL}?slug=${slug}`,
    });
  }

  return entries;
}

export function cleanEntryMarkdown(md: string): string {
  return md
    .replace(/\{%.*?%\}/gs, "")
    .replace(/<figure>|<\/figure>/g, "")
    .replace(/<figcaption>.*?<\/figcaption>/gs, "")
    .replace(/<details>/g, "\n---\n")
    .replace(/<\/details>/g, "")
    .replace(/<summary>(.*?)<\/summary>/g, "### $1");
}
