import React from "react";
import UGCAgentBlock from "../UGCAgentBlock";
import { usePagination } from "@/hooks/usePagination";
import ErrorState from "../ErrorState";
import {
  Block,
  BlockUIType,
  LibraryAgent,
  SpecialBlockID,
} from "@/lib/autogpt-server-api";
import { useBlockMenuContext } from "../block-menu-provider";

const MyAgentsContent: React.FC = () => {
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

  const handleAddAgent = (agent: LibraryAgent) => {
    const block = {
      id: SpecialBlockID.AGENT,
      name: agent.name,
      description:
        `Ver.${agent.graph_version}` +
        (agent.description ? ` | ${agent.description}` : ""),
      categories: [{ category: "AGENT", description: "" }],
      inputSchema: agent.input_schema,
      outputSchema: agent.output_schema,
      staticOutput: false,
      uiType: BlockUIType.AGENT,
      uiKey: agent.id,
      costs: [],
      hardcodedValues: {
        graph_id: agent.graph_id,
        graph_version: agent.graph_version,
        input_schema: agent.input_schema,
        output_schema: agent.output_schema,
      },
    } as Block;

    addNode(block);
  };

  if (loading) {
    return (
      <div
        ref={scrollRef}
        className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 transition-all duration-200 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200"
      >
        <div className="w-full space-y-3 px-4 pb-4">
          {[0, 1, 2, 3, 4].map((index) => (
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
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 transition-all duration-200 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200"
    >
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <UGCAgentBlock
            key={agent.id}
            title={agent.name}
            edited_time={agent.updated_at}
            version={agent.graph_version}
            image_url={agent.image_url}
            onClick={() => handleAddAgent(agent)}
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

export default MyAgentsContent;
