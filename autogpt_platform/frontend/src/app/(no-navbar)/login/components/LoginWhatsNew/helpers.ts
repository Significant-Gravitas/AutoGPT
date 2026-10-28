import type { ChangelogEntry } from "@/components/molecules/ChangelogPopup/helpers";

export const RELEASE_COUNT = 3;

// Past this the panel links to the changelog instead of waiting on it.
export const CHANGELOG_TIMEOUT_MS = 5000;

export interface Release {
  slug: string;
  url: string;
  dateRange: string;
  headline: string;
  details: string;
}

// A release's highlights are one comma-separated markdown cell: the first
// heads the release and the rest are its details.
export function toRelease(entry: ChangelogEntry): Release {
  const [first = "", ...rest] = splitHighlights(toPlainText(entry.highlights));
  return {
    slug: entry.slug,
    url: entry.url,
    dateRange: toPlainText(entry.dateRange),
    headline: capitalize(first) || "Release notes",
    details: capitalize(rest.join(", ")),
  };
}

export function getReleaseLabel(release: Release, isLatest: boolean) {
  const latest = isLatest ? ", latest release" : "";
  return `${release.headline}, ${release.dateRange}${latest} (opens in a new tab)`;
}

export function toPlainText(markdown: string): string {
  return (
    markdown
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
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

function capitalize(text: string) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
