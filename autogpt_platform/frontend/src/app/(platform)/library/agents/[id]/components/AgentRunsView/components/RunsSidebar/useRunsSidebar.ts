"use client";

import { useEffect, useMemo, useState } from "react";

import { useGetV1ListGraphExecutionsInfinite } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionsPaginated } from "@/app/api/__generated__/models/graphExecutionsPaginated";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useSearchParams } from "next/navigation";

const AGENT_RUNNING_POLL_INTERVAL = 1500;

type Args = {
  graphId?: string;
  onSelectRun: (runId: string) => void;
};

export function useRunsSidebar({ graphId, onSelectRun }: Args) {
  const params = useSearchParams();
  const existingRunId = params.get("executionId") as string | undefined;
  const [tabValue, setTabValue] = useState<"runs" | "scheduled">("runs");

  const runsQuery = useGetV1ListGraphExecutionsInfinite(
    graphId || "",
    { page: 1, page_size: 20 },
    {
      query: {
        enabled: !!graphId,
        // Lightweight polling so statuses refresh; only poll if any run is active
        refetchInterval: (q) => {
          if (tabValue !== "runs") return false;
          const pages = q.state.data?.pages as
            | Array<{ data: unknown }>
            | undefined;
          if (!pages || pages.length === 0) return false;
          try {
            const executions = pages.flatMap((p) => {
              const response = p.data as GraphExecutionsPaginated;
              return response.executions || [];
            });
            const hasActive = executions.some(
              (e: { status?: string }) =>
                e.status === "RUNNING" || e.status === "QUEUED",
            );
            return hasActive ? AGENT_RUNNING_POLL_INTERVAL : false;
          } catch {
            return false;
          }
        },
        refetchIntervalInBackground: true,
        refetchOnWindowFocus: false,
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

  useEffect(() => {
    if (runs.length > 0) {
      if (existingRunId) {
        onSelectRun(existingRunId);
        return;
      }
      onSelectRun(runs[0].id);
    }
  }, [runs, existingRunId]);

  useEffect(() => {
    if (existingRunId && existingRunId.startsWith("schedule:"))
      setTabValue("scheduled");
    else setTabValue("runs");
  }, [existingRunId]);

  const schedules: GraphExecutionJobInfo[] =
    schedulesQuery.data?.status === 200 ? schedulesQuery.data.data : [];

  // If there are no runs but there are schedules, and nothing is selected, auto-select the first schedule
  useEffect(() => {
    if (!existingRunId && runs.length === 0 && schedules.length > 0)
      onSelectRun(`schedule:${schedules[0].id}`);
  }, [existingRunId, runs.length, schedules, onSelectRun]);

  return {
    runs,
    schedules,
    error: schedulesQuery.error || runsQuery.error,
    loading: !schedulesQuery.isSuccess || !runsQuery.isSuccess,
    runsQuery,
    tabValue,
    setTabValue,
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
