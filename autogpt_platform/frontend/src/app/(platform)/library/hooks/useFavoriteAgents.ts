"use client";

import { useGetV2ListFavoriteLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useMemo } from "react";
import { filterAgents } from "../components/LibraryAgentList/helpers";

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
    },
    {
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const allAgents = agentsQueryData
    ? unpaginate(agentsQueryData, "agents")
    : [];

  const filteredAgents = useMemo(
    () => filterAgents(allAgents, searchTerm),
    [allAgents, searchTerm],
  );

  const agentCount = filteredAgents.length;

  return {
    allAgents: filteredAgents,
    agentLoading,
    hasNextPage,
    agentCount,
    isFetchingNextPage,
    fetchNextPage,
  };
}
