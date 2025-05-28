import React, { Fragment } from "react";
import BlocksList from "./BlocksList";
import Block from "../Block";
import { BlockRequest } from "@/lib/autogpt-server-api";
import { usePagination } from "@/hooks/usePagination";
import ErrorState from "../ErrorState";

interface PaginatedBlocksContentProps {
  blockRequest: BlockRequest;
  pageSize?: number;
}

const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
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

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {[0, 1, 3].map((categoryIndex) => (
          <Fragment key={categoryIndex}>
            {categoryIndex > 0 && (
              <div className="my-4 h-[1px] w-full bg-zinc-100" />
            )}
            {[0, 1, 2].map((blockIndex) => (
              <Block.Skeleton key={`${categoryIndex}-${blockIndex}`} />
            ))}
          </Fragment>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full px-4 pb-4">
        <ErrorState
          title="Failed to load blocks"
          error={error}
          onRetry={refresh}
        />
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 transition-all duration-200 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200"
    >
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

export default PaginatedBlocksContent;
