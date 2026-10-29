import {
  CHANGELOG_BASE_URL,
  CHANGELOG_INDEX_MD_URL,
} from "./changelog-constants";

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

// The published releases, newest first. Rejects when the request fails, so
// each caller decides what an unavailable changelog looks like.
export async function fetchChangelogEntries(
  signal?: AbortSignal,
): Promise<ChangelogEntry[]> {
  const response = await fetch(CHANGELOG_INDEX_MD_URL, { signal });
  if (!response.ok) {
    throw new Error(`Changelog request failed with ${response.status}`);
  }
  return parseChangelogIndex(await response.text());
}

// A release the account is newer than holds nothing new for that user.
export function isReleaseBeforeAccount(
  entry: ChangelogEntry,
  accountCreatedAt: string | undefined,
): boolean {
  const lastDay = getReleaseLastDay(entry.slug);
  const createdAt = accountCreatedAt ? Date.parse(accountCreatedAt) : NaN;
  if (lastDay === null || Number.isNaN(createdAt)) return false;
  return createdAt >= lastDay.getTime() + DAY_MS;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const MONTHS = [
  "january",
  "february",
  "march",
  "april",
  "may",
  "june",
  "july",
  "august",
  "september",
  "october",
  "november",
  "december",
];

// Slugs end with the release's last day: "october-8-october-14-2026".
function getReleaseLastDay(slug: string): Date | null {
  const match = /([a-z]+)-(\d{1,2})-(\d{4})$/.exec(slug);
  if (!match) return null;
  const month = MONTHS.indexOf(match[1]);
  const [year, day] = [Number(match[3]), Number(match[2])];
  if (month === -1) return null;
  const date = new Date(Date.UTC(year, month, day));
  // Date.UTC rolls an impossible date over: "february-30" becomes March 2.
  const exists =
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month &&
    date.getUTCDate() === day;
  return exists ? date : null;
}
