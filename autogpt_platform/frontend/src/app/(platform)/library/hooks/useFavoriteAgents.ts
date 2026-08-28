"use client";

import {
  getGetV2ListFavoriteLibraryAgentsQueryKey,
  useGetV2ListFavoriteLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useMemo } from "react";
import { filterAgents } from "../components/LibraryAgentList/helpers";

interface Props {
  searchTerm: string;
  organizationId: string | null;
  teamId: string | null;
}

export function useFavoriteAgents({
  searchTerm,
  organizationId,
  teamId,
}: Props) {
  const params = {
    page: 1,
    page_size: 10,
  };
  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: agentLoading,
  } = useGetV2ListFavoriteLibraryAgentsInfinite(params, {
    query: {
      getNextPageParam: getPaginationNextPageNumber,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListFavoriteLibraryAgentsQueryKey(params),
        organizationId,
        teamId,
      ),
    },
    request: getTenantRequestInit(organizationId, teamId),
  });

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
