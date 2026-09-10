import type { ComponentType } from "react";

export interface SearchCommandItem {
  id: string;
  title: string;
  subtitle?: string | null;
  // Icon rendered at the start of the row. Accepts any component
  // accepting ``className`` — Phosphor / lucide / heroicons all match.
  // Library-specific ``aria-hidden`` / ``weight`` props vary in their
  // exact signatures, so this stays loose with ``ComponentType<{
  // className?: string }>``. Callers pass the icon component directly.
  icon?: ComponentType<{ className?: string }>;
}

export interface SearchCommandBucket {
  // Stable id for keys + ``onSelectItem`` callbacks.
  key: string;
  label: string;
  items: SearchCommandItem[];
}

export interface FlatBucketItem {
  item: SearchCommandItem;
  bucketKey: string;
  /** Absolute index across all buckets (0-based). */
  index: number;
}

export interface HighlightPart {
  text: string;
  isMatch: boolean;
}

export function flattenBuckets(
  buckets: SearchCommandBucket[],
): FlatBucketItem[] {
  const flat: FlatBucketItem[] = [];
  for (const bucket of buckets) {
    for (const item of bucket.items) {
      flat.push({ item, bucketKey: bucket.key, index: flat.length });
    }
  }
  return flat;
}

export function getTotalCount(buckets: SearchCommandBucket[]): number {
  let total = 0;
  for (const bucket of buckets) total += bucket.items.length;
  return total;
}

/**
 * Splits ``title`` so the substring matching ``query`` (case-insensitive)
 * is marked ``isMatch: true``. Returns the title unchanged when the query
 * is empty or no match is found.
 */
export function highlightMatch(title: string, query: string): HighlightPart[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  if (!normalizedQuery) return [{ text: title, isMatch: false }];

  const matchIndex = title.toLocaleLowerCase().indexOf(normalizedQuery);
  if (matchIndex === -1) return [{ text: title, isMatch: false }];

  const matchEnd = matchIndex + normalizedQuery.length;
  return [
    { text: title.slice(0, matchIndex), isMatch: false },
    { text: title.slice(matchIndex, matchEnd), isMatch: true },
    { text: title.slice(matchEnd), isMatch: false },
  ].filter((part) => part.text.length > 0);
}
