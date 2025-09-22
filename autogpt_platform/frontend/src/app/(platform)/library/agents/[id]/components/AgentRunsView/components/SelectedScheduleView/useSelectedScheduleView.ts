"use client";

import { useMemo } from "react";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

export function useSelectedScheduleView(graphId: string, scheduleId: string) {
  const query = useGetV1ListExecutionSchedulesForAGraph(graphId, {
    query: {
      enabled: !!graphId,
      select: (res) =>
        res.status === 200 ? (res.data as GraphExecutionJobInfo[]) : [],
    },
  });

  const schedule = useMemo(
    () => query.data?.find((s) => s.id === scheduleId),
    [query.data, scheduleId],
  );

  const httpError =
    query.isSuccess && !schedule
      ? { status: 404, statusText: "Not found" }
      : undefined;

  return {
    schedule,
    isLoading: query.isLoading,
    error: query.error || httpError,
  } as const;
}
