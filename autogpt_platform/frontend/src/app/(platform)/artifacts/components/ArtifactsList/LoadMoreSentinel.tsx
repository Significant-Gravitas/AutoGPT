"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { useEffect, useRef } from "react";

interface Props {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
}

export function LoadMoreSentinel({ hasMore, isLoading, onLoadMore }: Props) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const onLoadMoreRef = useRef(onLoadMore);
  onLoadMoreRef.current = onLoadMore;

  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore || isLoading) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) onLoadMoreRef.current();
      },
      { rootMargin: "400px 0px 400px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, isLoading]);

  if (!hasMore && !isLoading) return null;

  return (
    <div
      ref={sentinelRef}
      className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2 md:grid-cols-4"
      data-testid="artifacts-load-more-sentinel"
    >
      {isLoading
        ? Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-64 w-full rounded-2xl" />
          ))
        : null}
    </div>
  );
}
