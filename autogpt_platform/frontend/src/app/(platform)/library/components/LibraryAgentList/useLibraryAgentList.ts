"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { useLibraryPageContext } from "../state-provider";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export const useLibraryAgentList = () => {
  const isAgentFavoritingEnabled = useGetFlag(Flag.AGENT_FAVORITING);
  const { searchTerm, librarySort } = useLibraryPageContext();
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

  // Sort agents to put favorites first only if feature flag is enabled
  const sortedAgents = isAgentFavoritingEnabled
    ? [...allAgents].sort((a, b) => {
        // First priority: favorites
        if (a.is_favorite && !b.is_favorite) return -1;
        if (!a.is_favorite && b.is_favorite) return 1;

        // If both are favorites or both are not favorites, maintain original order
        return 0;
      })
    : allAgents;

  const agentCount = agents?.pages?.[0]
    ? (agents.pages[0].data as LibraryAgentResponse).pagination.total_items
    : 0;

  return {
    allAgents: sortedAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
};
