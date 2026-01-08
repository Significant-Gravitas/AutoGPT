"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import { useLibraryAgentsStore } from "@/hooks/useLibraryAgents/store";
import { useLibraryPageContext } from "../state-provider";
import { getInitialData } from "./helpers";

export function useLibraryAgentList() {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const { agents: cachedAgents } = useLibraryAgentsStore();

  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 8,
      search_term: searchTerm || undefined,
      sort_by: librarySort,
    },
    {
      query: {
        initialData: getInitialData(cachedAgents, searchTerm, 8),
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  const allAgents = agentsQueryData
    ? unpaginate(agentsQueryData, "agents")
    : [];
  const agentCount = getPaginatedTotalCount(agentsQueryData);

  return {
    allAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
}
