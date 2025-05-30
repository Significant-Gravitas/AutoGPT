import React from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import { usePagination } from "@/hooks/usePagination";
import ErrorState from "../ErrorState";

const MarketplaceAgentsContent: React.FC = () => {
  const {
    data: agents,
    loading,
    loadingMore,
    hasMore,
    error,
    scrollRef,
    refresh,
  } = usePagination({
    request: { apiType: "store-agents" },
    pageSize: 10,
  });

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {[0, 1, 2, 3, 4].map((index) => (
          <MarketplaceAgentBlock.Skeleton key={index} />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full p-4">
        <ErrorState
          title="Failed to load marketplace agents"
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
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <MarketplaceAgentBlock
            key={agent.slug}
            slug={agent.slug}
            title={agent.agent_name}
            image_url={agent.agent_image}
            creator_name={agent.creator}
            number_of_runs={agent.runs}
          />
        ))}
        {loadingMore && hasMore && (
          <>
            {Array.from({ length: 3 }).map((_, index) => (
              <MarketplaceAgentBlock.Skeleton key={`loading-${index}`} />
            ))}
          </>
        )}
      </div>
    </div>
  );
};

export default MarketplaceAgentsContent;
