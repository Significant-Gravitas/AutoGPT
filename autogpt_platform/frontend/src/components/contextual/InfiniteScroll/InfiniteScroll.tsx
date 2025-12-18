"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useInfiniteScroll } from "./useInfiniteScroll";
import LoadingBox from "@/components/__legacy__/ui/loading";

type InfiniteScrollProps = {
  children: React.ReactNode;
  hasNextPage: boolean;
  loader?: React.ReactNode;
  scrollThreshold?: number;
  className?: string;
  onLoadMore?: () => void;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  direction?: "vertical" | "horizontal";
} & (
  | {
      loadedItemsCount: number;
      endMessage: React.ReactNode;
    }
  | {
      loadedItemsCount?: never;
      endMessage?: never;
    }
);

export const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  loadedItemsCount,
  hasNextPage,
  loader,
  endMessage,
  className,
  scrollThreshold = 20,
  onLoadMore,
  isFetchingNextPage,
  fetchNextPage,
  direction = "vertical",
}) => {
  const { containerRef, endOfListRef } = useInfiniteScroll({
    isFetchingNextPage,
    fetchNextPage,
    scrollThreshold,
    onLoadMore,
    hasNextPage,
  });

  const defaultLoader = <LoadingBox className="w-full py-4" spinnerSize={12} />;

  return (
    <div
      ref={containerRef}
      className={cn(
        direction === "vertical" ? "w-full" : "flex h-full items-center",
        className,
      )}
    >
      {children}
      {hasNextPage ? (
        <div
          ref={endOfListRef}
          className={`flex items-center justify-center ${direction === "vertical" ? "w-full py-8" : "h-full flex-shrink-0 px-8"}`}
        >
          {loader || defaultLoader}
        </div>
      ) : (
        loadedItemsCount && loadedItemsCount > 0 && endMessage
      )}
    </div>
  );
};
