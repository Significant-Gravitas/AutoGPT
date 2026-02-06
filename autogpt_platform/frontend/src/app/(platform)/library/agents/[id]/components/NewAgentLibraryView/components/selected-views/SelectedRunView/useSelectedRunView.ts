"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { useEffect, useRef, useState } from "react";
import {
  EMPTY_EXECUTION_UPDATES_THRESHOLD,
  isEmptyExecutionUpdate,
  isPollingStatus,
} from "@/lib/executionPollingWatchdog";

export function useSelectedRunView(graphId: string, runId: string) {
  const emptyUpdatesCountRef = useRef(0);
  const stuckRef = useRef(false);
  const setStuckRef = useRef<((stuck: boolean) => void) | null>(null);
  const [executionStuck, setExecutionStuck] = useState(false);

  useEffect(() => {
    emptyUpdatesCountRef.current = 0;
    stuckRef.current = false;
    setExecutionStuck(false);
  }, [graphId, runId]);

  const executionQuery = useGetV1GetExecutionDetails(graphId, runId, {
    query: {
      refetchInterval: (q) => {
        if (stuckRef.current) return false;

        const rawData = q.state.data;
        if (!rawData) return false;

        if (isEmptyExecutionUpdate(rawData)) {
          emptyUpdatesCountRef.current += 1;
          if (
            emptyUpdatesCountRef.current >= EMPTY_EXECUTION_UPDATES_THRESHOLD
          ) {
            stuckRef.current = true;
            setStuckRef.current?.(true);
            return false;
          }
        } else {
          emptyUpdatesCountRef.current = 0;
          setStuckRef.current?.(false);
        }

        const status =
          (rawData as { status?: number }).status === 200
            ? (rawData as { data?: { status?: string } }).data?.status
            : undefined;

        if (!status) return false;
        if (isPollingStatus(status)) return 1500;
        return false;
      },
      refetchIntervalInBackground: true,
      refetchOnWindowFocus: false,
    },
  });

  setStuckRef.current = setExecutionStuck;

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
  } as const;
}
