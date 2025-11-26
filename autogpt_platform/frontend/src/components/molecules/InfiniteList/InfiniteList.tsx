"use client";

import React from "react";

interface InfiniteListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
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
    <div className={className}>
      {items.map((item, idx) => (
        <div key={idx} className={itemWrapperClassName}>
          {renderItem(item, idx)}
        </div>
      ))}
      <div ref={sentinelRef} />
    </div>
  );
}
