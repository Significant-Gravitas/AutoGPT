"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { useEffect, useRef } from "react";
import type { ArtifactsView } from "../../useArtifactsPage";
import { SkeletonRow } from "./ArtifactsTable/SkeletonRow";

interface Props {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  view: ArtifactsView;
}

export function LoadMoreSentinel({
  hasMore,
  isLoading,
  onLoadMore,
  view,
}: Props) {
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

  if (view === "list") {
    return (
      <div
        ref={sentinelRef}
        className="divide-y divide-zinc-100 border-t border-zinc-100"
        data-testid="artifacts-load-more-sentinel"
      >
        {isLoading
          ? Array.from({ length: 3 }).map((_, i) => <SkeletonRow key={i} />)
          : null}
      </div>
    );
  }

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
