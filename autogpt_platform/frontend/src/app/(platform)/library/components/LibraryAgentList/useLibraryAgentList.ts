"use client";

import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useEffect, useRef } from "react";

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
}

export function useLibraryAgentList({ searchTerm, librarySort }: Props) {
  const queryClient = getQueryClient();
  const prevSortRef = useRef<LibraryAgentSort | null>(null);

  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 20,
      search_term: searchTerm || undefined,
      sort_by: librarySort,
    },
    {
      query: {
        getNextPageParam: getPaginationNextPageNumber,
      },
    },
  );

  // Reset queries when sort changes to ensure fresh data with correct sorting
  useEffect(() => {
    if (prevSortRef.current !== null && prevSortRef.current !== librarySort) {
      // Reset all library agent queries to ensure fresh fetch with new sort
      queryClient.resetQueries({
        queryKey: ["/api/library/agents"],
      });
    }
    prevSortRef.current = librarySort;
  }, [librarySort, queryClient]);

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
