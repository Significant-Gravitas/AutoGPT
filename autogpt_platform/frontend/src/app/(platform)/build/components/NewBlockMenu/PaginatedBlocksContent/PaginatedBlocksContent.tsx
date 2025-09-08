import React from "react";
import { BlocksList } from "../BlockList/BlockList";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { usePaginatedBlocks } from "./usePaginatedBlocks";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { CircleNotchIcon } from "@phosphor-icons/react";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

interface PaginatedBlocksContentProps {
  type?: "all" | "input" | "action" | "output" | null;
}

export const PaginatedBlocksContent: React.FC<PaginatedBlocksContentProps> = ({
  type,
}) => {
  const {
    allBlocks: blocks,
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
        responseError={error || undefined}
        context="blocks"
        onRetry={() => refetch()}
        /> 
      </div>
     
    );
  }

  return (
    <div className={cn(scrollbarStyles, "h-full overflow-y-auto pt-4 transition-all duration-200")}>
        <InfiniteScroll
          isFetchingNextPage={isFetchingNextPage}
          fetchNextPage={fetchNextPage}
          hasNextPage={hasNextPage}
          loader={<CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />}
        >
          <BlocksList blocks={blocks} loading={blocksLoading} />
        </InfiniteScroll>
    </div>
  );
};