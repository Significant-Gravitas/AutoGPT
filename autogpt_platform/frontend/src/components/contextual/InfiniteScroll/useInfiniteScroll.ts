import { environment } from "@/services/environment";
import { useCallback, useEffect, useRef, useState } from "react";

interface useInfiniteScrollProps {
  scrollThreshold: number;
  onLoadMore?: () => void;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  hasNextPage: boolean;
}

export const useInfiniteScroll = ({
  onLoadMore,
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
  scrollThreshold,
}: useInfiniteScrollProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const endOfListRef = useRef<HTMLDivElement>(null);
  const [isInView, setIsInView] = useState(false);
  const isLoadingRef = useRef(false);

  const loadMore = useCallback(async () => {
    if (hasNextPage && !isFetchingNextPage && !isLoadingRef.current) {
      isLoadingRef.current = true;
      try {
        fetchNextPage();
        onLoadMore?.();
      } finally {
        isLoadingRef.current = false;
      }
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, onLoadMore]);

  useEffect(() => {
    if (!hasNextPage || !endOfListRef.current || environment.isServerSide())
      return;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        setIsInView(entry.isIntersecting);
      },
      {
        rootMargin: `${scrollThreshold}px`,
      },
    );

    observer.observe(endOfListRef.current);

    return () => {
      observer.disconnect();
    };
  }, [hasNextPage, scrollThreshold]);

  useEffect(() => {
    if (isInView && hasNextPage && !isLoadingRef.current) {
      loadMore();
    }
  }, [isInView, hasNextPage]);

  return {
    containerRef,
    endOfListRef,
  };
};
