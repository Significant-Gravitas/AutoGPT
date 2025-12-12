"use client";

import { useMemo } from "react";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";

export function useSelectedScheduleView(graphId: string, scheduleId: string) {
  const schedulesQuery = useGetV1ListExecutionSchedulesForAGraph(graphId, {
    query: {
      enabled: !!graphId,
      select: okData,
    },
  });

  const schedule = useMemo(
    () => schedulesQuery.data?.find((s) => s.id === scheduleId),
    [schedulesQuery.data, scheduleId],
  );

  const httpError =
    schedulesQuery.isSuccess && !schedule
      ? { status: 404, statusText: "Not found" }
      : undefined;

  return {
    schedule,
    isLoading: schedulesQuery.isLoading,
    error: schedulesQuery.error || httpError,
  } as const;
}
