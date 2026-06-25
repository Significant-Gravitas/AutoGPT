import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import {
  flattenSessions,
  SESSION_LIST_QUERY_KEY,
  type SessionListInfiniteData,
} from "./useSessionList";

const TITLE_POLL_INTERVAL_MS = 2_000;
const TITLE_POLL_MAX_ATTEMPTS = 5;

interface Args {
  sessionId: string | null;
  status: string;
  isReconnecting: boolean;
}

/**
 * After a chat stream completes, the backend generates a title asynchronously.
 * Poll the session list until the new title appears (or give up) so the sidebar
 * animates in without a manual refresh.
 */
export function useSessionTitlePoll({
  sessionId,
  status,
  isReconnecting,
}: Args) {
  const queryClient = useQueryClient();
  const titlePollRef = useRef<ReturnType<typeof setInterval>>();
  const prevStatusRef = useRef(status);

  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isNowReady = status === "ready";

    if (!wasActive || !isNowReady || !sessionId || isReconnecting) return;

    queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });

    const sid = sessionId;
    let attempts = 0;
    clearInterval(titlePollRef.current);
    titlePollRef.current = setInterval(() => {
      const data = queryClient.getQueryData<SessionListInfiniteData>(
        SESSION_LIST_QUERY_KEY,
      );
      const hasTitle = flattenSessions(data).some(
        (s) => s.id === sid && s.title,
      );
      if (hasTitle || attempts >= TITLE_POLL_MAX_ATTEMPTS) {
        clearInterval(titlePollRef.current);
        titlePollRef.current = undefined;
        return;
      }
      attempts += 1;
      queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
    }, TITLE_POLL_INTERVAL_MS);
  }, [status, sessionId, isReconnecting, queryClient]);

  useEffect(() => {
    return () => {
      clearInterval(titlePollRef.current);
      titlePollRef.current = undefined;
    };
  }, [sessionId]);
}
