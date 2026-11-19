import { useStableCallback } from "@/hooks/useStableCallback";
import { useEffect, useRef } from "react";

/** How far above the list's end the next page starts loading. */
const LOAD_AHEAD_PX = 120;

interface Args {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
}

/** Loads the next page when the end of a scrolling list comes near. Once a
 *  page lands the observer is made again, so a list still too short to
 *  scroll keeps filling. */
export function useLoadMoreOnScroll({ hasMore, isLoading, onLoadMore }: Args) {
  const listRef = useRef<HTMLUListElement>(null);
  const sentinelRef = useRef<HTMLLIElement>(null);
  const loadMore = useStableCallback(onLoadMore);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore || isLoading) return;
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) loadMore();
      },
      {
        root: listRef.current,
        rootMargin: `0px 0px ${LOAD_AHEAD_PX}px 0px`,
      },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, isLoading, loadMore]);

  return { listRef, sentinelRef };
}
