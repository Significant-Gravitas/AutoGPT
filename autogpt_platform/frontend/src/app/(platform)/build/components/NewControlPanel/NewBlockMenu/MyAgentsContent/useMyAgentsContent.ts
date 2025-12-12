import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
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
    data: agentsQueryData,
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
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const allAgents = agentsQueryData
    ? unpaginate(agentsQueryData, "agents")
    : [];
  const status = agentsQueryData?.pages[0]?.status;

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
