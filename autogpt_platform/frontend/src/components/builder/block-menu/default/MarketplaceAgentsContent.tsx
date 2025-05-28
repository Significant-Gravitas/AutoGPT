import React from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import { Button } from "@/components/ui/button";
import { usePagination } from "@/hooks/usePagination";

const MarketplaceAgentsContent: React.FC = () => {
  const { data: agents, loading, loadingMore, hasMore, error, scrollRef, refresh } = usePagination({
    request: { apiType: 'store-agents' },
    pageSize: 10,
  });

  return (
    <div 
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 transition-all duration-200"
    >
      <div className="w-full space-y-3 px-4 pb-4">
        {loading
          ? Array(5)
              .fill(null)
              .map((_, index) => (
                <MarketplaceAgentBlock.Skeleton key={index} />
              ))
          : agents.map((agent) => (
              <MarketplaceAgentBlock
                key={agent.slug}
                title={agent.agent_name}
                image_url={agent.agent_image}
                creator_name={agent.creator}
                number_of_runs={agent.runs}
              />
            ))}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3">
            <p className="text-sm text-red-600 mb-2">
              Error loading marketplace agents: {error}
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
        )}
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
