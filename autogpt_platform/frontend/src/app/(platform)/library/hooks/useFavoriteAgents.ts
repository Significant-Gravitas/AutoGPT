"use client";

import { useGetV2ListFavoriteLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";

interface Props {
  searchTerm: string;
}

export function useFavoriteAgents({ searchTerm }: Props) {
  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useGetV2ListFavoriteLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 10,
      search_term: searchTerm || undefined,
    },
    {
      query: { getNextPageParam: getPaginationNextPageNumber },
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
