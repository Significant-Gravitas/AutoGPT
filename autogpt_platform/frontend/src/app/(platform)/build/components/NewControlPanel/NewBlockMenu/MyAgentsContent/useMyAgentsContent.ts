import {
  getV2GetLibraryAgent,
  useGetV2ListLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useState } from "react";
import { convertLibraryAgentIntoCustomNode } from "../helpers";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useShallow } from "zustand/react/shallow";
import { useReactFlow } from "@xyflow/react";

export const useMyAgentsContent = () => {
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [isGettingAgentDetails, setIsGettingAgentDetails] = useState(false);
  const addBlock = useNodeStore(useShallow((state) => state.addBlock));
  const { setViewport } = useReactFlow();
  // This endpoints is not giving info about inputSchema and outputSchema
  // Will create new endpoint for this
  const {
    data: agents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isError,
    isLoading: agentLoading,
    refetch,
    error,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 10,
    },
    {
      query: {
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as LibraryAgentResponse).pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const allAgents =
    agents?.pages?.flatMap((page) => {
      const response = page.data as LibraryAgentResponse;
      return response.agents;
    }) ?? [];

  const status = agents?.pages[0]?.status;

  const handleAddBlock = async (agent: LibraryAgent) => {
    setSelectedAgentId(agent.id);
    setIsGettingAgentDetails(true);

    try {
      const response = await getV2GetLibraryAgent(agent.id);

      if (!response.data) {
        console.error("Failed to get agent details", selectedAgentId, agent.id);
        return;
      }

      const { input_schema, output_schema } = response.data as LibraryAgent;
      const { block, hardcodedValues } = convertLibraryAgentIntoCustomNode(
        agent,
        input_schema,
        output_schema,
      );
      const customNode = addBlock(block, hardcodedValues);
      setTimeout(() => {
        setViewport(
          {
            x: -customNode.position.x * 0.8 + window.innerWidth / 2,
            y: -customNode.position.y * 0.8 + (window.innerHeight - 400) / 2,
            zoom: 0.8,
          },
          { duration: 500 },
        );
      }, 50);
    } catch (error) {
      console.error("Error adding block:", error);
    } finally {
      setSelectedAgentId(null);
      setIsGettingAgentDetails(false);
    }
  };

  return {
    allAgents,
    agentLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    isError,
    refetch,
    error,
    status,
    handleAddBlock,
    isGettingAgentDetails,
    selectedAgentId,
  };
};
