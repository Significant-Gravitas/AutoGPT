"use client";

import {
  postV2CreateSession,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMobileDrawer } from "./components/MobileDrawer/useMobileDrawer";
import { useSessionsPagination } from "./components/SessionsList/useSessionsPagination";
import {
  checkReadyToShowContent,
  filterVisibleSessions,
  getCurrentSessionId,
  mergeCurrentSessionIntoList,
  shouldAutoSelectSession,
} from "./helpers";

export function useCopilotShell() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const {
    isDrawerOpen,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
  } = useMobileDrawer();

  const paginationEnabled = !isMobile || isDrawerOpen;

  const {
    sessions: accumulatedSessions,
    isLoading: isSessionsLoading,
    isFetching: isSessionsFetching,
    hasNextPage,
    areAllSessionsLoaded,
    totalCount,
    fetchNextPage,
    reset: resetPagination,
  } = useSessionsPagination({
    enabled: paginationEnabled,
  });

  const storedSessionId = storage.get(Key.CHAT_SESSION_ID) ?? null;
  const currentSessionId = useMemo(
    () => getCurrentSessionId(searchParams, storedSessionId),
    [searchParams, storedSessionId],
  );

  const { data: currentSessionData, isLoading: isCurrentSessionLoading } =
    useGetV2GetSession(currentSessionId || "", {
      query: {
        enabled: !!currentSessionId && paginationEnabled,
        select: okData,
      },
    });

  const [hasAutoSelectedSession, setHasAutoSelectedSession] = useState(false);
  const hasCreatedSessionRef = useRef(false);
  const paramSessionId = searchParams.get("sessionId");

  const createSessionAndNavigate = useCallback(
    function createSessionAndNavigate() {
      postV2CreateSession({ body: JSON.stringify({}) })
        .then((response) => {
          if (response.status === 200 && response.data) {
            router.push(`/copilot/chat?sessionId=${response.data.id}`);
            setHasAutoSelectedSession(true);
          }
        })
        .catch(() => {
          hasCreatedSessionRef.current = false;
        });
    },
    [router],
  );

  useEffect(() => {
    if (!areAllSessionsLoaded || hasAutoSelectedSession) return;

    const visibleSessions = filterVisibleSessions(accumulatedSessions);
    const autoSelect = shouldAutoSelectSession(
      areAllSessionsLoaded,
      hasAutoSelectedSession,
      paramSessionId,
      visibleSessions,
      accumulatedSessions,
      isSessionsLoading,
      totalCount,
    );

    if (paramSessionId) {
      setHasAutoSelectedSession(true);
      return;
    }

    if (autoSelect.shouldSelect && autoSelect.sessionIdToSelect) {
      setHasAutoSelectedSession(true);
      router.push(`/copilot/chat?sessionId=${autoSelect.sessionIdToSelect}`);
    } else if (autoSelect.shouldCreate && !hasCreatedSessionRef.current) {
      hasCreatedSessionRef.current = true;
      createSessionAndNavigate();
    } else if (totalCount === 0) {
      setHasAutoSelectedSession(true);
    }
  }, [
    areAllSessionsLoaded,
    accumulatedSessions,
    paramSessionId,
    hasAutoSelectedSession,
    router,
    isSessionsLoading,
    totalCount,
    createSessionAndNavigate,
  ]);

  useEffect(() => {
    if (paramSessionId) {
      setHasAutoSelectedSession(true);
    }
  }, [paramSessionId]);

  function resetAutoSelect() {
    setHasAutoSelectedSession(false);
    hasCreatedSessionRef.current = false;
  }

  // Reset pagination and auto-selection when query becomes disabled
  useEffect(() => {
    if (!paginationEnabled) {
      resetPagination();
      resetAutoSelect();
    }
  }, [paginationEnabled, resetPagination]);

  const sessions = useMemo(
    function getSessions() {
      return mergeCurrentSessionIntoList(
        accumulatedSessions,
        currentSessionId,
        currentSessionData,
      );
    },
    [accumulatedSessions, currentSessionId, currentSessionData],
  );

  function handleSelectSession(sessionId: string) {
    router.push(`/copilot/chat?sessionId=${sessionId}`);
    if (isMobile) handleCloseDrawer();
  }

  function handleNewChat() {
    storage.clean(Key.CHAT_SESSION_ID);
    resetAutoSelect();
    createSessionAndNavigate();
    if (isMobile) handleCloseDrawer();
  }

  const isReadyToShowContent = useMemo(
    () =>
      checkReadyToShowContent(
        areAllSessionsLoaded,
        paramSessionId,
        accumulatedSessions,
        isCurrentSessionLoading,
        currentSessionData,
        hasAutoSelectedSession,
      ),
    [
      areAllSessionsLoaded,
      paramSessionId,
      accumulatedSessions,
      isCurrentSessionLoading,
      currentSessionData,
      hasAutoSelectedSession,
    ],
  );

  return {
    isMobile,
    isDrawerOpen,
    isLoading: isSessionsLoading || !areAllSessionsLoaded,
    sessions,
    currentSessionId,
    handleSelectSession,
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
