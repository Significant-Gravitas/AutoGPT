import type { MutableRefObject } from "react";
import {
  flattenBuckets,
  type FlatBucketItem,
  type SearchCommandBucket,
  type SearchCommandItem,
} from "./helpers";
import { SearchCommandResultItem } from "./SearchCommandResultItem";

interface Props {
  buckets: SearchCommandBucket[];
  idPrefix: string;
  query: string;
  highlightedIndex: number;
  highlightedRef: MutableRefObject<HTMLButtonElement | null>;
  onHighlight: (index: number) => void;
  onSelect: (item: SearchCommandItem, bucketKey: string) => void;
  /** Id of the row whose action is in-flight (renders a spinner). */
  loadingItemId?: string;
}

export function SearchCommandResults({
  buckets,
  idPrefix,
  query,
  highlightedIndex,
  highlightedRef,
  onHighlight,
  onSelect,
  loadingItemId,
}: Props) {
  const flat = flattenBuckets(buckets);
  // Group the flat list back by bucket so the absolute ``index`` survives
  // for keyboard nav while we still render each section under its
  // header.
  const grouped = new Map<string, FlatBucketItem[]>();
  for (const bucket of buckets) grouped.set(bucket.key, []);
  for (const entry of flat) grouped.get(entry.bucketKey)?.push(entry);

  return (
    <div
      role="listbox"
      aria-label="Search results"
      data-slot="search-command-results"
      className="flex flex-col gap-1"
    >
      {buckets.map((bucket) => {
        const entries = grouped.get(bucket.key) ?? [];
        if (entries.length === 0) return null;
        return (
          <div
            key={bucket.key}
            data-slot="search-command-results-group"
            className="px-2"
          >
            <div
              data-slot="search-command-results-group-label"
              className="px-3 pb-1 pt-2 text-caption-sm font-semibold uppercase leading-inherit tracking-wide text-foreground-tertiary"
            >
              {bucket.label}
            </div>
            <div
              data-slot="search-command-results-group-list"
              className="flex flex-col"
            >
              {entries.map((entry) => (
                <SearchCommandResultItem
                  key={entry.item.id}
                  item={entry.item}
                  idPrefix={idPrefix}
                  query={query}
                  isHighlighted={entry.index === highlightedIndex}
                  isLoading={entry.item.id === loadingItemId}
                  highlightedRef={
                    entry.index === highlightedIndex
                      ? highlightedRef
                      : undefined
                  }
                  onHighlight={() => onHighlight(entry.index)}
                  onSelect={() => onSelect(entry.item, entry.bucketKey)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
