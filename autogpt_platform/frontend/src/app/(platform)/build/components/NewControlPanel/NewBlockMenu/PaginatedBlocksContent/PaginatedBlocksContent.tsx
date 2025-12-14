import React from "react";
import { BlocksList } from "../BlockList/BlockList";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { usePaginatedBlocks } from "./usePaginatedBlocks";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";

interface PaginatedBlocksContentProps {
  type?: "all" | "input" | "action" | "output" | null;
}

export const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
  type,
}) => {
  const {
    allBlocks: blocks,
    status,
    blocksLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    refetch,
  } = usePaginatedBlocks({
    type,
  });

  if (error) {
    return (
      <div className="h-full px-4">
        <ErrorCard
          isSuccess={false}
          httpError={{
            status: status,
            statusText: "Request failed",
            message: (error?.detail as string) || "An error occurred",
          }}
          responseError={error || undefined}
          context="block menu"
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  return (
    <InfiniteScroll
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
      hasNextPage={hasNextPage}
      className={blockMenuContainerStyle}
    >
      <BlocksList blocks={blocks} loading={blocksLoading} />
    </InfiniteScroll>
  );
};
