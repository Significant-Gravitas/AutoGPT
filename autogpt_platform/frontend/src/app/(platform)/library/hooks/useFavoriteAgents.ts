"use client";

import { keepPreviousData } from "@tanstack/react-query";
import {
  getGetV2ListFavoriteLibraryAgentsQueryKey,
  useGetV2ListFavoriteLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

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
    search_term: searchTerm || undefined,
  };
  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isPlaceholderData,
  } = useGetV2ListFavoriteLibraryAgentsInfinite(params, {
    query: {
      getNextPageParam: getPaginationNextPageNumber,
      placeholderData: keepPreviousData,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListFavoriteLibraryAgentsQueryKey(params),
        organizationId,
        teamId,
      ),
    },
    request: getTenantRequestInit(organizationId, teamId),
  });

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
