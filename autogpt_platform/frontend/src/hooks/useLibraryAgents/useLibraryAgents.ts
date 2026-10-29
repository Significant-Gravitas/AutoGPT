import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useMemo } from "react";
import { buildAgentInfoMap } from "./store";

export function useLibraryAgents() {
  const { data: agentsQueryData, isLoading: isRefreshing } =
    useGetV2ListLibraryAgentsInfinite(
      {
        page: 1,
        page_size: 100,
        is_hidden: false,
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

  // Stable between renders, so consumers keyed on it (status maps, the
  // execution event subscription) only recompute when the data changes.
  const agents = useMemo(
    () => (agentsQueryData ? unpaginate(agentsQueryData, "agents") : []),
    [agentsQueryData],
  );

  // Use agents.length as dependency to avoid recreating map unnecessarily
  const agentInfoMap = useMemo(
    () => buildAgentInfoMap(agents),
    [agents.length, agents.map((a) => a.id).join(",")],
  );

  return { agents, agentInfoMap, isRefreshing, lastUpdatedAt: undefined };
}
