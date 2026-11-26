import { refetchAfterReconnect } from "@/hooks/useExecutionEvents";
import { useStableCallback } from "@/hooks/useStableCallback";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { type QueryKey, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

interface Args {
  queryKey: QueryKey;
  onTasks: boolean;
}

/**
 * Updates published while the socket was down never arrive, so re-read the
 * runs after a reconnect: at once on Tasks, or else once Tasks opens. The
 * read replaces a first page still loading, which may predate the reconnect.
 */
export function useRunsReadAfterReconnect({ queryKey, onTasks }: Args) {
  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const missedUpdatesRef = useRef(false);

  const readIfMissed = useStableCallback(() => {
    if (!onTasks || !missedUpdatesRef.current) return;
    missedUpdatesRef.current = false;
    refetchAfterReconnect(queryClient, queryKey);
  });

  useEffect(
    () =>
      api.onWebSocketReconnect(() => {
        missedUpdatesRef.current = true;
        readIfMissed();
      }),
    [api, readIfMissed],
  );
  useEffect(() => readIfMissed(), [onTasks, readIfMissed]);
}
