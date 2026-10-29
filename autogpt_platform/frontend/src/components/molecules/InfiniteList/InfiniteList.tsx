"use client";

import React from "react";

interface InfiniteListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  /** A stable identity per item, so rows keep their state when items are
   *  inserted above them. Falls back to the index. */
  getKey?: (item: T, index: number) => React.Key;
  onEndReached: () => void;
  hasMore: boolean;
  isFetchingMore?: boolean;
  className?: string;
  itemWrapperClassName?: string;
}

export function InfiniteList<T>(props: InfiniteListProps<T>) {
  const {
    items,
    renderItem,
    getKey,
    onEndReached,
    hasMore,
    isFetchingMore,
    className,
    itemWrapperClassName,
  } = props;
  const sentinelRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (!hasMore) return;

    const node = sentinelRef.current;
    if (!node) return;

    const observer = new IntersectionObserver((entries) => {
      const entry = entries[0];
      if (entry.isIntersecting && hasMore && !isFetchingMore) onEndReached();
    });

    observer.observe(node);

    return () => observer.disconnect();
  }, [hasMore, isFetchingMore, onEndReached]);

  return (
    <div data-slot="infinite-list" className={className}>
      {items.map((item, idx) => (
        <div
          key={getKey ? getKey(item, idx) : idx}
          data-slot="infinite-list-item"
          className={itemWrapperClassName}
        >
          {renderItem(item, idx)}
        </div>
      ))}
      <div ref={sentinelRef} data-slot="infinite-list-sentinel" />
    </div>
  );
}
