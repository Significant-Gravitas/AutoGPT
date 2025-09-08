import React from "react";
import { UGCAgentBlock } from "../UGCAgentBlock";
import { useMyAgentsContent } from "./useMyAgentsContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { blockMenuContainerStyle } from "../style";

export const MyAgentsContent = () => {
  const {
    allAgents,
    agentLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    isError,
    refetch,
  } = useMyAgentsContent();

  if (agentLoading) {
    return (
      <div className={blockMenuContainerStyle}>
        {Array.from({ length: 5 }).map((_, index) => (
          <UGCAgentBlock.Skeleton key={index} />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          context="library agents"
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
      {allAgents.map((agent) => (
        <UGCAgentBlock
          key={agent.id}
          title={agent.name}
          edited_time={agent.updated_at}
          version={agent.graph_version}
          image_url={agent.image_url}
        />
      ))}
    </InfiniteScroll>
  );
};
