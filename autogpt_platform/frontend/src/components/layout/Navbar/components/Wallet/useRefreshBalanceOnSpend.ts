import type { GraphExecution } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useEffect, useRef } from "react";

export const BALANCE_REFRESH_DELAY_MS = 1000;

const FINISHED_RUN_STATUSES: GraphExecution["status"][] = [
  "COMPLETED",
  "FAILED",
  "TERMINATED",
];

/**
 * Re-reads the balance after something may have spent from it: one of the
 * user's runs finishing, or a chat turn ending (a turn can run paid blocks).
 * A burst of events costs one read.
 */
export function useRefreshBalanceOnSpend(fetchCredits: () => void) {
  const api = useBackendAPI();
  const fetchCreditsRef = useRef(fetchCredits);

  useEffect(() => {
    fetchCreditsRef.current = fetchCredits;
  });

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;

    function scheduleRefresh() {
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        fetchCreditsRef.current();
      }, BALANCE_REFRESH_DELAY_MS);
    }

    // Listens without subscribing: run events reach this page only for the
    // workflows something on it already watches.
    const detachRuns = api.onWebSocketMessage(
      "graph_execution_event",
      (execution) => {
        if (FINISHED_RUN_STATUSES.includes(execution.status)) scheduleRefresh();
      },
    );
    const detachTurns = api.onWebSocketMessage(
      "notification",
      (notification) => {
        if (notification.type === "copilot_completion") scheduleRefresh();
      },
    );

    return () => {
      detachRuns();
      detachTurns();
      if (timer !== null) clearTimeout(timer);
    };
  }, [api]);
}
