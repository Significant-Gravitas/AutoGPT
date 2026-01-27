import { getGetV2ListSessionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { useChatStore } from "@/components/contextual/Chat/chat-store";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { useSessionsPagination } from "./components/SessionsList/useSessionsPagination";
import {
  convertSessionDetailToSummary,
  filterVisibleSessions,
  mergeCurrentSessionIntoList,
} from "./helpers";

interface UseShellSessionListArgs {
  paginationEnabled: boolean;
  currentSessionId: string | null;
  currentSessionData: SessionDetailResponse | null | undefined;
  isOnHomepage: boolean;
  paramSessionId: string | null;
}

export function useShellSessionList({
  paginationEnabled,
  currentSessionId,
  currentSessionData,
  isOnHomepage,
  paramSessionId,
}: UseShellSessionListArgs) {
  const queryClient = useQueryClient();
  const onStreamComplete = useChatStore((s) => s.onStreamComplete);

  const {
    sessions: accumulatedSessions,
    isLoading: isSessionsLoading,
    isFetching: isSessionsFetching,
    hasNextPage,
    fetchNextPage,
    reset: resetPagination,
  } = useSessionsPagination({
    enabled: paginationEnabled,
  });

  const recentlyCreatedSessionsRef = useRef<
    Map<string, SessionSummaryResponse>
  >(new Map());

  useEffect(() => {
    if (isOnHomepage && !paramSessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });
    }
  }, [isOnHomepage, paramSessionId, queryClient]);

  useEffect(() => {
    if (currentSessionId && currentSessionData) {
      const isNewSession =
        currentSessionData.updated_at === currentSessionData.created_at;
      const isNotInAccumulated = !accumulatedSessions.some(
        (s) => s.id === currentSessionId,
      );
      if (isNewSession || isNotInAccumulated) {
        const summary = convertSessionDetailToSummary(currentSessionData);
        recentlyCreatedSessionsRef.current.set(currentSessionId, summary);
      }
    }
  }, [currentSessionId, currentSessionData, accumulatedSessions]);

  useEffect(() => {
    for (const sessionId of recentlyCreatedSessionsRef.current.keys()) {
      if (accumulatedSessions.some((s) => s.id === sessionId)) {
        recentlyCreatedSessionsRef.current.delete(sessionId);
      }
    }
  }, [accumulatedSessions]);

  useEffect(() => {
    const unsubscribe = onStreamComplete(() => {
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });
    });
    return unsubscribe;
  }, [onStreamComplete, queryClient]);

  const sessions = useMemo(
    () =>
      mergeCurrentSessionIntoList(
        accumulatedSessions,
        currentSessionId,
        currentSessionData,
        recentlyCreatedSessionsRef.current,
      ),
    [accumulatedSessions, currentSessionId, currentSessionData],
  );

  const visibleSessions = useMemo(
    () => filterVisibleSessions(sessions),
    [sessions],
  );

  const isLoading = isSessionsLoading && accumulatedSessions.length === 0;

  return {
    sessions: visibleSessions,
    isLoading,
    isSessionsFetching,
    hasNextPage,
    fetchNextPage,
    resetPagination,
    recentlyCreatedSessionsRef,
  };
}
