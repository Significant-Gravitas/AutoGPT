"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useInfiniteScroll } from "./useInfiniteScroll";

interface InfiniteScrollProps {
  children: React.ReactNode;
  dataLength: number;
  hasNextPage: boolean;
  loader?: React.ReactNode;
  endMessage?: React.ReactNode;
  scrollThreshold?: number;
  className?: string;
  scrollableTarget?: string;
  onLoadMore?: () => void;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
}

export const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  dataLength,
  hasNextPage,
  loader,
  endMessage,
  className,
  scrollThreshold = 20,
  scrollableTarget,
  onLoadMore,
  isFetchingNextPage,
  fetchNextPage,
}) => {
  const { containerRef, bottomRef } = useInfiniteScroll({
    isFetchingNextPage,
    fetchNextPage,
    scrollThreshold,
    scrollableTarget,
    onLoadMore,
    hasNextPage,
  });

  const defaultLoader = (
    <div className="flex w-full items-center justify-center py-4">
      <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
    </div>
  );

  return (
    <div ref={containerRef} className={cn("w-full", className)}>
      {children}
      {hasNextPage ? (
        <div
          ref={bottomRef}
          className="flex w-full items-center justify-center py-8"
        >
          {loader || defaultLoader}
        </div>
      ) : (
        dataLength > 0 && endMessage
      )}
    </div>
  );
};
