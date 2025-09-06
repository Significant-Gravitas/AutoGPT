import React from "react";
import { BlocksList } from "../BlockList/BlockList";
import { Block } from "../Block";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { usePaginatedBlocks } from "./usePaginatedBlocks";

interface PaginatedBlocksContentProps {
  category?: string | null;
  type?: "all" | "input" | "action" | "output" | null;
  provider?: string | null;
  pageSize?: number;
}

export const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
  category,
  type,
  provider,
  pageSize = 10,
}) => {
  const {
    allBlocks: blocks,
    blocksLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = usePaginatedBlocks({
    category,
    type,
    provider,
    pageSize,
  });

  const LoadingSpinner = () => (
    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-t-2 border-neutral-800" />
  );

  if (blocksLoading) {
    return (
      <div className="h-full w-full px-4 pb-4">
        <div className="flex flex-col items-center justify-center h-40 space-y-4">
          <p className="text-red-500">Failed to load blocks</p>
          <button
            onClick={() => fetchNextPage()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      {blocksLoading ? (
        <div className="flex h-[200px] items-center justify-center">
          <LoadingSpinner />
        </div>
      ) : (
        <InfiniteScroll
          isFetchingNextPage={isFetchingNextPage}
          fetchNextPage={fetchNextPage}
          hasNextPage={hasNextPage}
          loader={<LoadingSpinner />}
        >
          <BlocksList blocks={blocks} loading={false} />
        </InfiniteScroll>
      )}
    </div>
  );
};