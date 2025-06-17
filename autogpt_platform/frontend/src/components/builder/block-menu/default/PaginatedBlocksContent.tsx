import React, { Fragment } from "react";
import { BlocksList } from "./BlocksList";
import { Block } from "../Block";
import { BlockRequest } from "@/lib/autogpt-server-api";
import { usePagination } from "@/hooks/usePagination";
import { ErrorState } from "../ErrorState";
import { scrollbarStyles } from "@/components/styles/scrollbar";

interface PaginatedBlocksContentProps {
  blockRequest: BlockRequest;
  pageSize?: number;
}

export const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
  blockRequest,
  pageSize = 10,
}) => {
  const {
    data: blocks,
    loading,
    loadingMore,
    hasMore,
    error,
    scrollRef,
    refresh,
  } = usePagination({
    request: { apiType: "blocks", ...blockRequest },
    pageSize,
  });

  if (error) {
    return (
      <div className="h-full w-full px-4 pb-4">
        <ErrorState
          title="Failed to load blocks"
          error={error}
          onRetry={refresh}
        />
      </div>
    );
  }

  return (
    <div ref={scrollRef} className={scrollbarStyles}>
      <BlocksList blocks={blocks} loading={loading} />
      {loadingMore && hasMore && (
        <div className="w-full space-y-3 px-4 pb-4">
          {Array.from({ length: 3 }).map((_, index) => (
            <Block.Skeleton key={`loading-${index}`} />
          ))}
        </div>
      )}
    </div>
  );
};
