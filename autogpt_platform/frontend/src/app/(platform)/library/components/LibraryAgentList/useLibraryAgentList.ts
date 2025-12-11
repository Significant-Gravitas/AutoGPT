"use client";

import { getPaginationNextPageNumber } from "@/app/api/helpers";
import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useLibraryPageContext } from "../state-provider";
import { useLibraryAgentsStore } from "@/hooks/useLibraryAgents/store";
import { getInitialData } from "./helpers";

export const useLibraryAgentList = () => {
  const { searchTerm, librarySort } = useLibraryPageContext();
  const { agents: cachedAgents } = useLibraryAgentsStore();

  const {
    data: agents,
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

  const allAgents =
    agents?.pages?.flatMap((page) => {
      const response = page.data as LibraryAgentResponse;
      return response.agents;
    }) ?? [];

  const agentCount = agents?.pages?.[0]
    ? (agents.pages[0].data as LibraryAgentResponse).pagination.total_items
    : 0;

  return {
    allAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
};
