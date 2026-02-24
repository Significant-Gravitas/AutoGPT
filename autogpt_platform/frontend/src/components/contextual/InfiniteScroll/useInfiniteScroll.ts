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

    // Check if element is initially in view after a short delay to ensure DOM is ready
    const checkInitialView = () => {
      if (endOfListRef.current) {
        const rect = endOfListRef.current.getBoundingClientRect();
        const isInitiallyInView =
          rect.top <= window.innerHeight + scrollThreshold &&
          rect.bottom >= -scrollThreshold;

        if (isInitiallyInView) {
          setIsInView(true);
        }
      }
    };

    // Check immediately and after a short delay to catch cases where DOM updates
    checkInitialView();
    const timeoutId = setTimeout(checkInitialView, 100);

    return () => {
      clearTimeout(timeoutId);
      observer.disconnect();
    };
  }, [hasNextPage, scrollThreshold]);

  useEffect(() => {
    if (isInView && hasNextPage && !isLoadingRef.current) {
      loadMore();
    }
  }, [isInView, hasNextPage, loadMore]);

  return {
    containerRef,
    endOfListRef,
  };
};
