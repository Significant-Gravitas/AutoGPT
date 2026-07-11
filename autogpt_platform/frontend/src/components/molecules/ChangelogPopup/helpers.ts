import {
  CHANGELOG_BASE_URL,
  CHANGELOG_IMAGE_PROXY_URL,
  CHANGELOG_PROXY_URL,
} from "./changelog-constants";

export interface ChangelogEntry {
  slug: string;
  dateRange: string;
  highlights: string;
  url: string;
  mdUrl: string;
}

export function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
  // gitbook README rows: | [May 7 – June 10](may-7-june-10-2026.md) | ... |
  const rowPattern = /\|\s*\[([^\]]+)\]\(([a-z0-9-]+)\.md\)\s*\|\s*([^|]+)\|/g;

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
  return (
    md
      // Drop GitBook's export boilerplate blockquote.
      .replace(/^>\s*For the complete documentation index.*$/gim, "")
      // GitBook <figure> images point at ../.gitbook/assets/<file>; route them
      // through our cached image proxy and turn them into markdown images so
      // react-markdown renders them.
      .replace(
        /<figure>\s*<img\s+src="[^"]*\.gitbook\/assets\/([^"?]+)"(?:\s+alt="([^"]*)")?[^>]*>\s*(?:<figcaption>\s*(?:<p>)?([\s\S]*?)(?:<\/p>)?\s*<\/figcaption>)?\s*<\/figure>/g,
        (_m, file, alt, caption) => {
          const src = `${CHANGELOG_IMAGE_PROXY_URL}?file=${encodeURIComponent(file)}`;
          const image = `![${(alt ?? caption ?? "").trim()}](${src})`;
          return caption ? `${image}\n\n*${caption.trim()}*\n` : image;
        },
      )
      // Any remaining GitBook-isms.
      .replace(/\{%.*?%\}/gs, "")
      .replace(/<figure>|<\/figure>/g, "")
      .replace(/<figcaption>.*?<\/figcaption>/gs, "")
      .replace(/<details>/g, "\n---\n")
      .replace(/<\/details>/g, "")
      .replace(/<summary>(.*?)<\/summary>/g, "### $1")
  );
}
