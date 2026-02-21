import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useEffect, useMemo } from "react";
import { buildAgentInfoMap } from "./store";

export function useLibraryAgents() {
  const {
    data: agentsQueryData,
    isLoading: isRefreshing,
    hasNextPage,
    fetchNextPage,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 100,
    },
    {
      query: {
        getNextPageParam: getPaginationNextPageNumber,
        // Don't block rendering - fetch in background
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 5 * 60 * 1000, // 5 minutes
      },
    },
  );

  // Automatically fetch all pages so every agent appears in the map
  useEffect(() => {
    if (hasNextPage) {
      fetchNextPage();
    }
  }, [hasNextPage, fetchNextPage]);

  const agents = agentsQueryData ? unpaginate(agentsQueryData, "agents") : [];

  // Use agents.length as dependency to avoid recreating map unnecessarily
  const agentInfoMap = useMemo(
    () => buildAgentInfoMap(agents),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [agents.length, agents.map((a) => a.id).join(",")],
  );

  return { agents, agentInfoMap, isRefreshing, lastUpdatedAt: undefined };
}
