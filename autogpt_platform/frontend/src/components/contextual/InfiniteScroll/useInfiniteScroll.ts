import { isServerSide } from "@/lib/utils/is-server-side";
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
  const [isLoading, setIsLoading] = useState(false);

  const handleLoadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const loadMore = useCallback(async () => {
    if (hasNextPage && !isLoading) {
      setIsLoading(true);
      try {
        handleLoadMore();
        onLoadMore?.();
      } finally {
        setIsLoading(false);
      }
    }
  }, [hasNextPage, isLoading, handleLoadMore, onLoadMore]);

  useEffect(() => {
    if (!hasNextPage || !endOfListRef.current || isServerSide()) return;

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
    if (isInView && hasNextPage && !isLoading) {
      loadMore();
    }
  }, [isInView, hasNextPage, isLoading, loadMore]);

  return {
    containerRef,
    endOfListRef,
  };
};
