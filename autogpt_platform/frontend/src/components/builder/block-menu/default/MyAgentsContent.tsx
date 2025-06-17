import React from "react";
import { UGCAgentBlock } from "../UGCAgentBlock";
import { usePagination } from "@/hooks/usePagination";
import { ErrorState } from "../ErrorState";
import { useBlockMenuContext } from "../block-menu-provider";
import { convertLibraryAgentIntoBlock } from "@/lib/utils";

export const MyAgentsContent = () => {
  const {
    data: agents,
    loading,
    loadingMore,
    hasMore,
    error,
    scrollRef,
    refresh,
  } = usePagination({
    request: { apiType: "library-agents" },
    pageSize: 10,
  });
  const { addNode } = useBlockMenuContext();

  if (loading) {
    return (
      <div
        ref={scrollRef}
        className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 h-full overflow-y-auto pt-4 transition-all duration-200"
      >
        <div className="w-full space-y-3 px-4 pb-4">
          {Array.from({ length: 5 }).map((_, index) => (
            <UGCAgentBlock.Skeleton key={index} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full p-4">
        <ErrorState
          title="Failed to load library agents"
          error={error}
          onRetry={refresh}
        />
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 h-full overflow-y-auto pt-4 transition-all duration-200"
    >
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <UGCAgentBlock
            key={agent.id}
            title={agent.name}
            edited_time={agent.updated_at}
            version={agent.graph_version}
            image_url={agent.image_url}
            onClick={() => {
              const block = convertLibraryAgentIntoBlock(agent);
              addNode(block);
            }}
          />
        ))}
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
