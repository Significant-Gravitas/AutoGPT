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
      className="flex flex-col gap-1"
    >
      {buckets.map((bucket) => {
        const entries = grouped.get(bucket.key) ?? [];
        if (entries.length === 0) return null;
        return (
          <div key={bucket.key} className="px-2">
            <div className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
              {bucket.label}
            </div>
            <div className="flex flex-col">
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
                  // Absolute index across all buckets so the stagger
                  // cascades naturally section-by-section instead of
                  // restarting at each header.
                  enterIndex={entry.index}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
