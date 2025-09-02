"use client";

import { useEffect, useMemo } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionsPaginated } from "@/app/api/__generated__/models/graphExecutionsPaginated";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

export function useRunsSidebar(graphId?: string) {
  const runsQuery = useGetV1ListGraphExecutionsInfinite(
    graphId || "",
    { page: 1, page_size: 20 },
    {
      query: {
        enabled: !!graphId,
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as GraphExecutionsPaginated)
            .pagination;
          const hasMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return hasMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(
    graphId || "",
    {
      query: { enabled: !!graphId },
    },
  );

  const runs = useMemo(
    () =>
      runsQuery.data?.pages.flatMap((p) => {
        const response = p.data as GraphExecutionsPaginated;
        return response.executions;
      }) || [],
    [runsQuery.data],
  );

  const schedules: GraphExecutionJobInfo[] =
    schedulesQuery.data?.status === 200 ? schedulesQuery.data.data : [];

  return {
    runs,
    schedules,
    error: schedulesQuery.error || runsQuery.error,
    loading: !schedulesQuery.isSuccess || !runsQuery.isSuccess,
    runsQuery,
    runsCount:
      (
        runsQuery.data?.pages.at(-1)?.data as
          | GraphExecutionsPaginated
          | undefined
      )?.pagination.total_items || runs.length,
    schedulesCount: schedules.length,
    fetchMoreRuns: runsQuery.fetchNextPage,
    hasMoreRuns: runsQuery.hasNextPage,
    isFetchingMoreRuns: runsQuery.isFetchingNextPage,
  };
}
