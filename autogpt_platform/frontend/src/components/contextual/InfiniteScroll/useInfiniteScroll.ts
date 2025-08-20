import { isServerSide } from "@/lib/utils/is-server-side";
import { debounce } from "lodash";
import { useCallback, useEffect, useRef, useState } from "react";

interface useInfiniteScrollProps {
  scrollThreshold: number;
  scrollableTarget?: string;
  onLoadMore?: () => void;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  hasNextPage: boolean;
  direction?: "vertical" | "horizontal";
}

export const useInfiniteScroll = ({
  scrollableTarget,
  onLoadMore,
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
  scrollThreshold,
  direction = "vertical",
}: useInfiniteScrollProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [isInView, setIsInView] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleScroll = useCallback(() => {
    if (containerRef.current && !isServerSide()) {
      const container = containerRef.current;
      const containerRect = container.getBoundingClientRect();

      let isVisible: boolean;

      if (direction === "horizontal") {
        const { right } = containerRect;
        const { innerWidth } = window;
        isVisible = right <= innerWidth + scrollThreshold;
      } else {
        const { bottom } = containerRect;
        const { innerHeight } = window;
        isVisible = bottom <= innerHeight + scrollThreshold;
      }

      setIsInView(isVisible);
    }
  }, [scrollThreshold, direction]);

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
    if (!hasNextPage) return;

    const handleDebouncedScroll = debounce(handleScroll, 200);

    const scrollElement = scrollableTarget
      ? document.querySelector(scrollableTarget)
      : window;

    if (!scrollElement) return;

    scrollElement.addEventListener("scroll", handleDebouncedScroll);

    handleScroll();

    return () => {
      scrollElement.removeEventListener("scroll", handleDebouncedScroll);
    };
  }, [handleScroll, hasNextPage, scrollableTarget]);

  useEffect(() => {
    if (isInView && hasNextPage && !isLoading) {
      loadMore();
    }
  }, [isInView, hasNextPage, isLoading, loadMore]);

  return {
    containerRef,
    bottomRef,
  };
};
