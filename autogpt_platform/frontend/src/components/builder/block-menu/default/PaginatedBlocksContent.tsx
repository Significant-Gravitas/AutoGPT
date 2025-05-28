import React from "react";
import BlocksList from "./BlocksList";
import Block from "../Block";
import { Button } from "@/components/ui/button";
import { BlockRequest } from "@/lib/autogpt-server-api";
import { usePagination } from "@/hooks/usePagination";

interface PaginatedBlocksContentProps {
  blockRequest: BlockRequest;
  pageSize?: number;
}

const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
  blockRequest,
  pageSize = 10,
}) => {
  const { data: blocks, loading, loadingMore, hasMore, error, scrollRef, refresh } = usePagination({
    request: { apiType: 'blocks', ...blockRequest },
    pageSize,
  });

  return (
    <div 
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 transition-all duration-200"
    >
      <BlocksList blocks={blocks} loading={loading} />
      {error && (
        <div className="w-full px-4 pb-4">
          <div className="rounded-lg border border-red-200 bg-red-50 p-3">
            <p className="text-sm text-red-600 mb-2">
              Error loading blocks: {error}
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              className="h-7 text-xs"
            >
              Retry
            </Button>
          </div>
        </div>
      )}
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