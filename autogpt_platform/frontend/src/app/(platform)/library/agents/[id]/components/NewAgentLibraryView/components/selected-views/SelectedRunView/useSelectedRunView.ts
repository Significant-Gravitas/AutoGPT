"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { useExecutionPollingWatchdog } from "@/lib/useExecutionPollingWatchdog";
import { useRef } from "react";

export function useSelectedRunView(graphId: string, runId: string) {
  const refetchRef = useRef<(() => void) | null>(null);

  const { executionStuck, clearStuckAndRetry, refetchInterval } =
    useExecutionPollingWatchdog({
      refetch: () => refetchRef.current?.(),
      resetKey: `${graphId}:${runId}`,
    });

  const executionQuery = useGetV1GetExecutionDetails(graphId, runId, {
    query: {
      refetchInterval,
      refetchIntervalInBackground: true,
      refetchOnWindowFocus: false,
    },
  });

  refetchRef.current = executionQuery.refetch;

  const run = okData(executionQuery.data);
  const status = executionQuery.data?.status;

  const presetId = run?.preset_id || undefined;

  const presetQuery = useGetV2GetASpecificPreset(presetId || "", {
    query: {
      enabled: !!presetId,
      select: okData,
    },
  });

  const httpError =
    status && status !== 200
      ? { status, statusText: `Request failed: ${status}` }
      : undefined;

  return {
    run,
    preset: presetQuery.data,
    isLoading: executionQuery.isLoading || presetQuery.isLoading,
    responseError: executionQuery.error || presetQuery.error,
    httpError,
    executionStuck,
    clearStuckAndRetry,
  } as const;
}
