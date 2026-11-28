"use client";

import { keepPreviousData } from "@tanstack/react-query";
import { useGetV2ListFavoriteLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "@/app/api/helpers";
import { hasUnknownSchedules } from "@/services/schedules/scheduler-availability";
import { useSchedulerProbe } from "@/services/schedules/useSchedulerProbe";
import { useEffect } from "react";

interface Props {
  searchTerm: string;
}

export function useFavoriteAgents({ searchTerm }: Props) {
  const {
    data: agentsQueryData,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
    isLoading,
    isPlaceholderData,
    refetch,
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

  // While the schedules are unknown, a cheap probe — one read of the user's
  // schedule list — checks whether the scheduler is back, instead of
  // re-reading every loaded page; once it answers, this list re-reads once per answer while its schedules stay unknown.
  const { recovered } = useSchedulerProbe(hasUnknownSchedules(allAgents));
  useEffect(() => {
    // cancelRefetch: false lets an in-flight multi-page reread finish: a
    // new probe answer while one runs is ignored instead of restarting it.
    if (recovered) void refetch({ cancelRefetch: false });
  }, [recovered, refetch]);

  const agentCount = getPaginatedTotalCount(agentsQueryData);

  return {
    allAgents,
    agentLoading,
    hasNextPage: hasNextPage && !isPlaceholderData,
    agentCount,
    isFetching,
    isFetchingNextPage,
    fetchNextPage,
  };
}
