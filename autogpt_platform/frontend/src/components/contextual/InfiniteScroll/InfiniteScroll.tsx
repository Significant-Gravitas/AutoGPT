"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useInfiniteScroll } from "./useInfiniteScroll";

type InfiniteScrollProps = {
  children: React.ReactNode;
  hasNextPage: boolean;
  loader?: React.ReactNode;
  scrollThreshold?: number;
  className?: string;
  scrollableTarget?: string;
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
  scrollableTarget,
  onLoadMore,
  isFetchingNextPage,
  fetchNextPage,
  direction = "vertical",
}) => {
  const { containerRef, bottomRef } = useInfiniteScroll({
    isFetchingNextPage,
    fetchNextPage,
    scrollThreshold,
    scrollableTarget,
    onLoadMore,
    hasNextPage,
    direction,
  });

  const defaultLoader = (
    <div className="flex w-full items-center justify-center py-4">
      <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
    </div>
  );

  return (
    <div
      ref={containerRef}
      className={cn(direction === "vertical" ? "w-full" : "h-full", className)}
    >
      {children}
      {hasNextPage ? (
        <div
          ref={bottomRef}
          className={`flex items-center justify-center ${direction === "vertical" ? "w-full py-8" : "h-full px-8"}`}
        >
          {loader || defaultLoader}
        </div>
      ) : (
        loadedItemsCount && loadedItemsCount > 0 && endMessage
      )}
    </div>
  );
};
