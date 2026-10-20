import { CHANGELOG_BASE_URL } from "./changelog-constants";

export interface ChangelogEntry {
  slug: string;
  dateRange: string;
  highlights: string;
  // The public docs page for this release.
  url: string;
}

export function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
  const rowPattern =
    /\|\s*\[([^\]]+)\]\(\/docs\/platform\/changelog\/changelog\/([a-z0-9-]+)\.md\)\s*\|\s*([^|]+)\|/g;

  let match;
  while ((match = rowPattern.exec(md)) !== null) {
    const [, dateRange, slug, highlights] = match;
    entries.push({
      slug,
      dateRange: dateRange.trim(),
      highlights: highlights.trim(),
      url: `${CHANGELOG_BASE_URL}/${slug}`,
    });
  }

  return entries;
}
