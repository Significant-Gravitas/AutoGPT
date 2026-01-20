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
import { useEffect, useRef, useState } from "react";
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
    totalCount,
    fetchNextPage,
    reset: resetPagination,
  } = useSessionsPagination({
    enabled: paginationEnabled,
  });

  const storedSessionId = storage.get(Key.CHAT_SESSION_ID) ?? null;
  const currentSessionId = getCurrentSessionId(searchParams, storedSessionId)

  const { data: currentSessionData, isLoading: isCurrentSessionLoading } =
    useGetV2GetSession(currentSessionId || "", {
      query: {
        enabled: !!currentSessionId,
        select: okData,
      },
    });

  const [hasAutoSelectedSession, setHasAutoSelectedSession] = useState(false);
  const hasCreatedSessionRef = useRef(false);
  const hasAutoSelectedRef = useRef(false);
  const paramPrompt = searchParams.get("prompt");

  useEffect(() => {
    function runCreateSession() {
      postV2CreateSession({ body: JSON.stringify({}) })
        .then((response) => {
          if (response.status === 200 && response.data) {
            const promptParam = paramPrompt
              ? `&prompt=${encodeURIComponent(paramPrompt)}`
              : "";
            router.push(
              `/copilot/chat?sessionId=${response.data.id}${promptParam}`,
            );
            hasAutoSelectedRef.current = true;
            setHasAutoSelectedSession(true);
          }
        })
        .catch(() => {
          hasCreatedSessionRef.current = false;
        });
    }

    // Don't auto-select or auto-create sessions on homepage without an explicit sessionId
    if (isOnHomepage && !paramSessionId) {
      if (!hasAutoSelectedRef.current) {
        hasAutoSelectedRef.current = true;
        setHasAutoSelectedSession(true);
      }
      return;
    }

    if (!areAllSessionsLoaded || hasAutoSelectedRef.current) return;

    // If there's a prompt parameter, create a new session (don't auto-select existing)
    if (paramPrompt && !paramSessionId && !hasCreatedSessionRef.current) {
      hasCreatedSessionRef.current = true;
      runCreateSession();
      return;
    }

    const visibleSessions = filterVisibleSessions(accumulatedSessions);
    
    const autoSelect = shouldAutoSelectSession(
      areAllSessionsLoaded,
      hasAutoSelectedRef.current,
      paramSessionId,
      visibleSessions,
      accumulatedSessions,
      isSessionsLoading,
      totalCount,
    );

    if (paramSessionId) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
      return;
    }

    // Don't auto-select existing sessions if there's a prompt (user wants new session)
    if (paramPrompt) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
      return;
    }

    if (autoSelect.shouldSelect && autoSelect.sessionIdToSelect) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
      router.push(`/copilot/chat?sessionId=${autoSelect.sessionIdToSelect}`);
    } else if (autoSelect.shouldCreate && !hasCreatedSessionRef.current) {
      // Only auto-create on chat page when no sessions exist, not homepage
      hasCreatedSessionRef.current = true;
      runCreateSession();
    } else if (totalCount === 0) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
    }
  }, [
    isOnHomepage,
    areAllSessionsLoaded,
    accumulatedSessions,
    paramSessionId,
    paramPrompt,
    router,
    isSessionsLoading,
    totalCount,
  ]);

  useEffect(() => {
    if (paramSessionId) {
      hasAutoSelectedRef.current = true;
      setHasAutoSelectedSession(true);
    }
  }, [paramSessionId]);

  // Reset pagination and auto-selection when query becomes disabled
  const prevPaginationEnabledRef = useRef(paginationEnabled);
  useEffect(() => {
    if (prevPaginationEnabledRef.current && !paginationEnabled) {
      resetPagination();
      resetAutoSelect();
    }
    prevPaginationEnabledRef.current = paginationEnabled;
  }, [paginationEnabled]);

  const sessions = mergeCurrentSessionIntoList(
    accumulatedSessions,
    currentSessionId,
    currentSessionData,
  );

  const sidebarSelectedSessionId =
    isOnHomepage && !paramSessionId ? null : currentSessionId;

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
      
  function handleSelectSession(sessionId: string) {
    router.push(`/copilot/chat?sessionId=${sessionId}`);
    if (isMobile) handleCloseDrawer();
  }

  function handleNewChat() {
    storage.clean(Key.CHAT_SESSION_ID);
    resetAutoSelect();
    router.push("/copilot");
    if (isMobile) handleCloseDrawer();
  }

  function resetAutoSelect() {
    hasAutoSelectedRef.current = false;
    setHasAutoSelectedSession(false);
    hasCreatedSessionRef.current = false;
  }

  return {
    isMobile,
    isDrawerOpen,
    isLoggedIn,
    hasActiveSession: Boolean(currentSessionId) && (!isOnHomepage || paramSessionId),
    isLoading: isSessionsLoading || !areAllSessionsLoaded,
    sessions,
    currentSessionId: sidebarSelectedSessionId,
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
