"use client";

import {
  postV2CreateSession,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Key, storage } from "@/services/storage/local-storage";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
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
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const breakpoint = useBreakpoint();
  const { isLoggedIn } = useSupabase();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";
  
  const isOnHomepage = pathname === "/copilot";

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
  const paramPrompt = searchParams.get("prompt");

  const createSessionAndNavigate = useCallback(
    function createSessionAndNavigate() {
      postV2CreateSession({ body: JSON.stringify({}) })
        .then((response) => {
          if (response.status === 200 && response.data) {
            const promptParam = paramPrompt ? `&prompt=${encodeURIComponent(paramPrompt)}` : "";
            router.push(`/copilot/chat?sessionId=${response.data.id}${promptParam}`);
            setHasAutoSelectedSession(true);
          }
        })
        .catch(() => {
          hasCreatedSessionRef.current = false;
        });
    },
    [router, paramPrompt],
  );

  useEffect(() => {
    // Don't auto-select or auto-create sessions on homepage
    if (isOnHomepage) {
      setHasAutoSelectedSession(true);
      return;
    }

    if (!areAllSessionsLoaded || hasAutoSelectedSession) return;

    // If there's a prompt parameter, create a new session (don't auto-select existing)
    if (paramPrompt && !paramSessionId && !hasCreatedSessionRef.current) {
      hasCreatedSessionRef.current = true;
      createSessionAndNavigate();
      return;
    }

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

    // Don't auto-select existing sessions if there's a prompt (user wants new session)
    if (paramPrompt) {
      setHasAutoSelectedSession(true);
      return;
    }

    if (autoSelect.shouldSelect && autoSelect.sessionIdToSelect) {
      setHasAutoSelectedSession(true);
      router.push(`/copilot/chat?sessionId=${autoSelect.sessionIdToSelect}`);
    } else if (autoSelect.shouldCreate && !hasCreatedSessionRef.current) {
      // Only auto-create on chat page when no sessions exist, not homepage
      hasCreatedSessionRef.current = true;
      createSessionAndNavigate();
    } else if (totalCount === 0) {
      setHasAutoSelectedSession(true);
    }
  }, [
    isOnHomepage,
    areAllSessionsLoaded,
    accumulatedSessions,
    paramSessionId,
    paramPrompt,
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

  const isReadyToShowContent = useMemo(() => {
    // On homepage, always show content (welcome screen) immediately
    if (isOnHomepage) return true;
    
    return checkReadyToShowContent(
      areAllSessionsLoaded,
      paramSessionId,
      accumulatedSessions,
      isCurrentSessionLoading,
      currentSessionData,
      hasAutoSelectedSession,
    );
  }, [
    isOnHomepage,
    areAllSessionsLoaded,
    paramSessionId,
    accumulatedSessions,
    isCurrentSessionLoading,
    currentSessionData,
    hasAutoSelectedSession,
  ]);

  return {
    isMobile,
    isDrawerOpen,
    isLoggedIn,
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
