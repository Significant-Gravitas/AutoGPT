"use client";

import type { Query } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  EMPTY_EXECUTION_UPDATES_THRESHOLD,
  EXECUTION_POLLING_INTERVAL_MS,
  isEmptyExecutionUpdate,
  isPollingStatus,
} from "@/lib/executionPollingWatchdog";

interface UseExecutionPollingWatchdogOptions {
  refetch: () => void;
  /** When these change, the empty-update count and stuck state reset. */
  resetKey?: string;
}

export function useExecutionPollingWatchdog({
  refetch,
  resetKey = "",
}: UseExecutionPollingWatchdogOptions) {
  const emptyUpdatesCountRef = useRef(0);
  const stuckRef = useRef(false);
  const setStuckRef = useRef<((stuck: boolean) => void) | null>(null);
  const [executionStuck, setExecutionStuck] = useState(false);

  useEffect(() => {
    emptyUpdatesCountRef.current = 0;
    stuckRef.current = false;
    setExecutionStuck(false);
  }, [resetKey]);

  setStuckRef.current = setExecutionStuck;

  function refetchInterval<TData>(
    query: Query<TData, Error, TData, readonly unknown[]>,
  ): number | false {
    if (stuckRef.current) return false;

    const rawData = query.state.data;
    if (!rawData) return false;

    if (isEmptyExecutionUpdate(rawData)) {
      emptyUpdatesCountRef.current += 1;
      if (emptyUpdatesCountRef.current >= EMPTY_EXECUTION_UPDATES_THRESHOLD) {
        stuckRef.current = true;
        setStuckRef.current?.(true);
        return false;
      }
      return EXECUTION_POLLING_INTERVAL_MS;
    }

    emptyUpdatesCountRef.current = 0;
    setStuckRef.current?.(false);

    const status =
      (rawData as { status?: number }).status === 200
        ? (rawData as { data?: { status?: string } }).data?.status
        : undefined;

    if (!status) return EXECUTION_POLLING_INTERVAL_MS;
    if (isPollingStatus(status)) return EXECUTION_POLLING_INTERVAL_MS;
    return false;
  }

  function clearStuckAndRetry() {
    stuckRef.current = false;
    setExecutionStuck(false);
    emptyUpdatesCountRef.current = 0;
    refetch();
  }

  return {
    executionStuck,
    clearStuckAndRetry,
    refetchInterval,
  } as const;
}
