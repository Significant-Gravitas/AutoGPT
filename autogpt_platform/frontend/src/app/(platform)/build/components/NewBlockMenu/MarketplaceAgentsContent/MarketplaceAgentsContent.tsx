import React from "react";
import { MarketplaceAgentBlock } from "../MarketplaceAgentBlock";
import { useMarketplaceAgentsContent } from "./useMarketplaceAgentsContent";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { CircleNotchIcon, EmptyIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { blockMenuContainerStyle } from "../style";

export const MarketplaceAgentsContent = () => {
  const { handleAddStoreAgent, addingAgent, isListStoreAgentsLoading, isListStoreAgentsError, listStoreAgentsError, listStoreAgents, refetchListStoreAgents, fetchNextPage, hasNextPage, isFetchingNextPage } = useMarketplaceAgentsContent();

  if (isListStoreAgentsLoading) {
    return (
      <div className={cn(scrollbarStyles, "h-full overflow-y-auto pt-4 transition-all duration-200")}>
        <div className="w-full space-y-3 px-4 pb-4">
          {Array.from({ length: 5 }).map((_, index) => (
            <MarketplaceAgentBlock.Skeleton key={index} />
          ))}
        </div>
      </div>
    );
  }

  if (isListStoreAgentsError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          context="marketplace agents"
          responseError={{detail: listStoreAgentsError?.detail}}
          onRetry={() => refetchListStoreAgents()}
        />
      </div>
    );
  }

  if (listStoreAgents.length === 0) {
    return (
      <div className="h-full p-4 flex flex-col items-center justify-center space-y-3">
        <EmptyIcon className="h-8 w-8 text-zinc-400" />
        <Text variant="body" >      
            No marketplace agents present
        </Text>
      </div>
    );
  }

  return (
      <InfiniteScroll
        isFetchingNextPage={isFetchingNextPage}
        fetchNextPage={fetchNextPage}
        hasNextPage={hasNextPage}
        loader={<CircleNotchIcon className="h-4 w-4 animate-spin" weight="bold" />}
        className={blockMenuContainerStyle}
      >
        {listStoreAgents?.map((agent, index) => (
          <MarketplaceAgentBlock
            key={agent.slug + index}
            slug={agent.slug}
            title={agent.agent_name}
            image_url={agent.agent_image}
            creator_name={agent.creator}
            number_of_runs={agent.runs}
            loading={addingAgent === agent.slug}
            onClick={() =>
              handleAddStoreAgent({
                creator_name: agent.creator,
                slug: agent.slug,
              })
            }
          />
        ))} 
      </InfiniteScroll>
     
  );
};