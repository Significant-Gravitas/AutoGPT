import {
  getGetV2ListSessionsQueryKey,
  type getV2ListSessionsResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

const TITLE_POLL_INTERVAL_MS = 2_000;
const TITLE_POLL_MAX_ATTEMPTS = 5;
const SESSION_LIST_LIMIT = 50;

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

    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey({ limit: SESSION_LIST_LIMIT }),
    });

    const sid = sessionId;
    let attempts = 0;
    clearInterval(titlePollRef.current);
    titlePollRef.current = setInterval(() => {
      const data = queryClient.getQueryData<getV2ListSessionsResponse>(
        getGetV2ListSessionsQueryKey({ limit: SESSION_LIST_LIMIT }),
      );
      const hasTitle =
        data?.status === 200 &&
        data.data.sessions.some((s) => s.id === sid && s.title);
      if (hasTitle || attempts >= TITLE_POLL_MAX_ATTEMPTS) {
        clearInterval(titlePollRef.current);
        titlePollRef.current = undefined;
        return;
      }
      attempts += 1;
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey({ limit: SESSION_LIST_LIMIT }),
      });
    }, TITLE_POLL_INTERVAL_MS);
  }, [status, sessionId, isReconnecting, queryClient]);

  useEffect(() => {
    return () => {
      clearInterval(titlePollRef.current);
      titlePollRef.current = undefined;
    };
  }, [sessionId]);
}
