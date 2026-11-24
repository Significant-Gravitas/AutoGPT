"use client";

import { keepPreviousData } from "@tanstack/react-query";
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
    isLoading,
    isPlaceholderData,
  } = useGetV2ListFavoriteLibraryAgentsInfinite(
    {
      page: 1,
      page_size: 10,
      search_term: searchTerm || undefined,
    },
    {
      query: {
        getNextPageParam: getPaginationNextPageNumber,
        placeholderData: keepPreviousData,
      },
    },
  );

  const allAgents =
    agentsQueryData && !isPlaceholderData
      ? unpaginate(agentsQueryData, "agents")
      : [];
  const agentLoading = isLoading || isPlaceholderData;

  const agentCount = getPaginatedTotalCount(agentsQueryData);

  return {
    allAgents,
    agentLoading,
    hasNextPage: hasNextPage && !isPlaceholderData,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
}
