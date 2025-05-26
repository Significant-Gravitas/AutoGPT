import React from "react";
import UGCAgentBlock from "../UGCAgentBlock";
import { Button } from "@/components/ui/button";
import { usePagination } from "@/hooks/usePagination";

const MyAgentsContent: React.FC = () => {
  const { data: agents, loading, loadingMore, hasMore, error, scrollRef, refresh } = usePagination({
    request: { apiType: 'library-agents' },
    pageSize: 10,
  });

  return (
    <div 
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
    >
      <div className="w-full space-y-3 px-4 pb-4">
        {loading
          ? Array(5)
              .fill(null)
              .map((_, index) => (
                <UGCAgentBlock.Skeleton key={index} />
              ))
          : agents.map((agent) => (
              <UGCAgentBlock
                key={agent.id}
                title={agent.name}
                edited_time={agent.updated_at}
                version={agent.graph_version}
                image_url={agent.image_url}
              />
            ))}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3">
            <p className="text-sm text-red-600 mb-2">
              Error loading library agents: {error}
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
              <UGCAgentBlock.Skeleton key={`loading-${index}`} />
            ))}
          </>
        )}
      </div>
    </div>
  );
};

export default MyAgentsContent;
