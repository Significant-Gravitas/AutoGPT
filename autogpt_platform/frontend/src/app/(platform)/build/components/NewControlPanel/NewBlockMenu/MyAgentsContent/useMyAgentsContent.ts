import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useAddAgentToBuilder } from "../hooks/useAddAgentToBuilder";
import { useToast } from "@/components/molecules/Toast/use-toast";

export const useMyAgentsContent = () => {
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [isGettingAgentDetails, setIsGettingAgentDetails] = useState(false);
  const { addLibraryAgentToBuilder } = useAddAgentToBuilder();
  const { toast } = useToast();

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
      await addLibraryAgentToBuilder(agent);
    } catch (error) {
      toast({
        title: "Failed to add agent to builder",
        description:
          ((error as any).message as string) || "An unexpected error occurred.",
        variant: "destructive",
      });
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
