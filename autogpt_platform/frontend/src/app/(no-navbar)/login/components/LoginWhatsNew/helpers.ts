import type { LatestRelease } from "@/app/api/changelog/helpers";

const LATEST_RELEASES_URL = "/api/changelog/latest";

// Past this the panel links to the changelog instead of waiting on it. The
// route's own upstream budget is at most half of this, leaving the rest for
// the trip from a slow connection to the route and back.
export const CHANGELOG_TIMEOUT_MS = 8000;

export interface Release {
  slug: string;
  url: string;
  dateRange: string;
  headline: string;
  details: string;
}

// Link URLs may hold one level of brackets: (https://x.org/a_(b)).
const MARKDOWN_LINK = /!?\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)/g;

// Stands in for a comma inside link text while the highlights are split.
const LINK_COMMA = "\uE000";

// The newest releases, each with its page title when the server could read it.
export async function fetchLatestReleases(
  signal: AbortSignal,
): Promise<LatestRelease[]> {
  const response = await fetch(LATEST_RELEASES_URL, { signal });
  if (!response.ok) {
    throw new Error(`Changelog request failed with ${response.status}`);
  }
  const { releases } = (await response.json()) as {
    releases: LatestRelease[];
  };
  return releases;
}

// A release heads with its page title over all of its highlights. Without a
// title, the first highlight heads it and the rest are its details.
export function toRelease(release: LatestRelease): Release {
  const title = toPlainText(release.title ?? "");
  const [first = "", ...rest] = toHighlights(release.highlights);
  const [headline, details] = title
    ? [title, [first, ...rest]]
    : [capitalize(first) || "Release notes", rest];
  return {
    slug: release.slug,
    url: release.url,
    dateRange: toPlainText(release.dateRange),
    headline,
    details: capitalize(details.filter(Boolean).join(", ")),
  };
}

export function getReleaseLabel(release: Release, isLatest: boolean) {
  const latest = isLatest ? ", latest release" : "";
  return `${release.headline}, ${release.dateRange}${latest} (opens in a new tab)`;
}

export function toPlainText(markdown: string): string {
  return (
    stripComments(markdown)
      .replace(MARKDOWN_LINK, "$1")
      .replace(/<br\s*\/?>/gi, " ")
      // An unclosed tag ("<script" at the end) goes too, not just "<b>…</b>".
      .replace(/<\/?[a-z][^>]*(?:>|$)/gi, "")
      .replace(/(\*\*|__|~~|`|\*)(\S(?:.*?\S)?)\1/g, "$2")
      .replace(/\\([\\`*_{}[\]()#+\-.!|~])/g, "$1")
      .replace(/&(amp|lt|gt|quot|#39|nbsp);/g, decodeEntity)
      .replace(/\s+/g, " ")
      .trim()
  );
}

// The highlights are one comma-separated markdown cell. Commas in link text
// ("[Clay, Rock](…)") are set aside first so they don't split a highlight.
function toHighlights(markdown: string) {
  const highlights = markdown.replace(MARKDOWN_LINK, (link) =>
    link.replaceAll(",", LINK_COMMA),
  );
  return splitHighlights(toPlainText(highlights)).map((highlight) =>
    highlight.replaceAll(LINK_COMMA, ","),
  );
}

// A comma inside brackets stays in its highlight: "New models (Claude, GPT)".
export function splitHighlights(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let current = "";
  for (const char of text) {
    if (char === "(" || char === "[") depth += 1;
    if ((char === ")" || char === "]") && depth > 0) depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  parts.push(current);
  return parts.map((part) => part.trim()).filter(Boolean);
}

const ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  "#39": "'",
  nbsp: " ",
};

function decodeEntity(_match: string, name: string) {
  return ENTITIES[name];
}

// An HTML comment, closed or cut off, never reaches the page.
function stripComments(text: string) {
  return text.replace(/<!--[\s\S]*?(?:-->|$)/g, "");
}

// Only an all-lowercase first word is capitalised, so "iOS" and "macOS"
// keep their casing.
function capitalize(text: string) {
  if (!/^[a-z]+(?![\w-])/.test(text)) return text;
  return text.charAt(0).toUpperCase() + text.slice(1);
}
