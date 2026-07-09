import { CHANGELOG_BASE_URL } from "./changelog-constants";

export interface ChangelogEntry {
  slug: string;
  dateRange: string;
  highlights: string;
  url: string;
  mdUrl: string;
}

export function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
  const rowPattern =
    /\|\s*\[([^\]]+)\]\((https?:\/\/[^)]+\/changelog\/changelog\/([a-z0-9-]+))\)\s*\|\s*([^|]+)\|/g;

  let match;
  while ((match = rowPattern.exec(md)) !== null) {
    const [, dateRange, url, slug, highlights] = match;
    entries.push({
      slug,
      dateRange: dateRange.trim(),
      highlights: highlights.trim(),
      url,
      mdUrl: `${CHANGELOG_BASE_URL}/${slug}.md`,
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
