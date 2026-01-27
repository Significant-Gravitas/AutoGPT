"use client";

import {
  getGetV2GetSessionQueryKey,
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { okData } from "@/app/api/helpers";
import { useChatStore } from "@/components/contextual/Chat/chat-store";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useCopilotStore } from "../../copilot-page-store";
import { useMobileDrawer } from "./components/MobileDrawer/useMobileDrawer";
import { useSessionsPagination } from "./components/SessionsList/useSessionsPagination";
import {
  checkReadyToShowContent,
  convertSessionDetailToSummary,
  filterVisibleSessions,
  getCurrentSessionId,
  mergeCurrentSessionIntoList,
} from "./helpers";

export function useCopilotShell() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const breakpoint = useBreakpoint();
  const { isLoggedIn } = useSupabase();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const [, setUrlSessionId] = useQueryState("sessionId", parseAsString);

  const isOnHomepage = pathname === "/copilot";
  const paramSessionId = searchParams.get("sessionId");

  const {
    isDrawerOpen,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
  } = useMobileDrawer();

  const paginationEnabled = !isMobile || isDrawerOpen || !!paramSessionId;

  const {
    sessions: accumulatedSessions,
    isLoading: isSessionsLoading,
    isFetching: isSessionsFetching,
    hasNextPage,
    areAllSessionsLoaded,
    fetchNextPage,
    reset: resetPagination,
  } = useSessionsPagination({
    enabled: paginationEnabled,
  });

  const currentSessionId = getCurrentSessionId(searchParams);

  const { data: currentSessionData, isLoading: isCurrentSessionLoading } =
    useGetV2GetSession(currentSessionId || "", {
      query: {
        enabled: !!currentSessionId,
        select: okData,
      },
    });

  const [hasAutoSelectedSession, setHasAutoSelectedSession] = useState(false);
  const hasAutoSelectedRef = useRef(false);
  const recentlyCreatedSessionsRef = useRef<
    Map<string, SessionSummaryResponse>
  >(new Map());

  const [optimisticSessionId, setOptimisticSessionId] = useState<string | null>(
    null,
  );

  useEffect(
    function clearOptimisticWhenUrlMatches() {
      if (optimisticSessionId && currentSessionId === optimisticSessionId) {
        setOptimisticSessionId(null);
      }
    },
    [currentSessionId, optimisticSessionId],
  );

  // Mark as auto-selected when sessionId is in URL
  useEffect(() => {
    if (paramSessionId && !hasAutoSelectedRef.current) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
    }
  }, [paramSessionId]);

  // On homepage without sessionId, mark as ready immediately
  useEffect(() => {
    if (isOnHomepage && !paramSessionId && !hasAutoSelectedRef.current) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
    }
  }, [isOnHomepage, paramSessionId]);

  // Invalidate sessions list when navigating to homepage (to show newly created sessions)
  useEffect(() => {
    if (isOnHomepage && !paramSessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });
    }
  }, [isOnHomepage, paramSessionId, queryClient]);

  // Track newly created sessions to ensure they stay visible even when switching away
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

  // Clean up recently created sessions that are now in the accumulated list
  useEffect(() => {
    for (const sessionId of recentlyCreatedSessionsRef.current.keys()) {
      if (accumulatedSessions.some((s) => s.id === sessionId)) {
        recentlyCreatedSessionsRef.current.delete(sessionId);
      }
    }
  }, [accumulatedSessions]);

  // Reset pagination when query becomes disabled
  const prevPaginationEnabledRef = useRef(paginationEnabled);
  useEffect(() => {
    if (prevPaginationEnabledRef.current && !paginationEnabled) {
      resetPagination();
      resetAutoSelect();
    }
    prevPaginationEnabledRef.current = paginationEnabled;
  }, [paginationEnabled, resetPagination]);

  const sessions = mergeCurrentSessionIntoList(
    accumulatedSessions,
    currentSessionId,
    currentSessionData,
    recentlyCreatedSessionsRef.current,
  );

  const visibleSessions = filterVisibleSessions(sessions);

  const sidebarSelectedSessionId =
    isOnHomepage && !paramSessionId && !optimisticSessionId
      ? null
      : optimisticSessionId || currentSessionId;

  const isReadyToShowContent = isOnHomepage
    ? true
    : checkReadyToShowContent(
        areAllSessionsLoaded,
        paramSessionId,
        accumulatedSessions,
        isCurrentSessionLoading,
        currentSessionData,
        hasAutoSelectedSession,
      );

  const stopStream = useChatStore((s) => s.stopStream);
  const onStreamComplete = useChatStore((s) => s.onStreamComplete);
  const setIsSwitchingSession = useCopilotStore((s) => s.setIsSwitchingSession);

  async function performSelectSession(sessionId: string) {
    if (sessionId === currentSessionId) return;

    const sourceSessionId = currentSessionId;

    if (sourceSessionId) {
      setIsSwitchingSession(true);

      await new Promise<void>(function waitForStreamComplete(resolve) {
        const unsubscribe = onStreamComplete(
          function handleComplete(completedId) {
            if (completedId === sourceSessionId) {
              clearTimeout(timeout);
              unsubscribe();
              resolve();
            }
          },
        );
        const timeout = setTimeout(function handleTimeout() {
          unsubscribe();
          resolve();
        }, 3000);
        stopStream(sourceSessionId);
      });

      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sourceSessionId),
      });
    }

    setOptimisticSessionId(sessionId);
    setUrlSessionId(sessionId, { shallow: false });
    setIsSwitchingSession(false);
    if (isMobile) handleCloseDrawer();
  }

  function handleSelectSession(sessionId: string) {
    if (sessionId === currentSessionId) return;
    setOptimisticSessionId(sessionId);
    setUrlSessionId(sessionId, { shallow: false });
    if (isMobile) handleCloseDrawer();
  }

  function handleNewChat() {
    resetAutoSelect();
    resetPagination();
    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey(),
    });
    setUrlSessionId(null, { shallow: false });
    setOptimisticSessionId(null);
    if (isMobile) handleCloseDrawer();
  }

  function resetAutoSelect() {
    hasAutoSelectedRef.current = false;
    setHasAutoSelectedSession(false);
  }

  const isLoading = isSessionsLoading && accumulatedSessions.length === 0;

  return {
    isMobile,
    isDrawerOpen,
    isLoggedIn,
    hasActiveSession:
      Boolean(currentSessionId) && (!isOnHomepage || Boolean(paramSessionId)),
    isLoading,
    sessions: visibleSessions,
    currentSessionId: sidebarSelectedSessionId,
    handleSelectSession,
    performSelectSession,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChat,
    hasNextPage,
    isFetchingNextPage: isSessionsFetching,
    fetchNextPage,
    isReadyToShowContent,
  };
}
