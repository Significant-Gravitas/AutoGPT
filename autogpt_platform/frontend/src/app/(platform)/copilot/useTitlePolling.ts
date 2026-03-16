import {
  getGetV2ListSessionsQueryKey,
  type getV2ListSessionsResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { getSessionListParams } from "./helpers";

const TITLE_POLL_INTERVAL_MS = 2_000;
const TITLE_POLL_MAX_ATTEMPTS = 5;

interface Props {
  isReconnecting: boolean;
  sessionId: string | null;
  status: string;
}

export function useTitlePolling({ isReconnecting, sessionId, status }: Props) {
  const queryClient = useQueryClient();
  const previousStatusRef = useRef(status);

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    previousStatusRef.current = status;

    const wasActive =
      previousStatus === "streaming" || previousStatus === "submitted";
    const isNowReady = status === "ready";

    if (!wasActive || !isNowReady || !sessionId || isReconnecting) {
      return;
    }

    const params = getSessionListParams();
    const queryKey = getGetV2ListSessionsQueryKey(params);
    let attempts = 0;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let isCancelled = false;

    const poll = () => {
      if (isCancelled) {
        return;
      }

      const data =
        queryClient.getQueryData<getV2ListSessionsResponse>(queryKey);
      const hasTitle =
        data?.status === 200 &&
        data.data.sessions.some(
          (session) => session.id === sessionId && session.title,
        );

      if (hasTitle || attempts >= TITLE_POLL_MAX_ATTEMPTS) {
        return;
      }

      attempts += 1;
      queryClient.invalidateQueries({ queryKey });
      timeoutId = setTimeout(poll, TITLE_POLL_INTERVAL_MS);
    };

    queryClient.invalidateQueries({ queryKey });
    timeoutId = setTimeout(poll, TITLE_POLL_INTERVAL_MS);

    return () => {
      isCancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [isReconnecting, queryClient, sessionId, status]);
}
