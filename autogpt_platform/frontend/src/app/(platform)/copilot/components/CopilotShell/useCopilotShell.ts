"use client";

import {
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useMobileDrawer } from "./components/MobileDrawer/useMobileDrawer";
import { useSessionsPagination } from "./components/SessionsList/useSessionsPagination";
import {
  checkReadyToShowContent,
  filterVisibleSessions,
  getCurrentSessionId,
  mergeCurrentSessionIntoList,
} from "./helpers";

export function useCopilotShell() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
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
  );

  const visibleSessions = filterVisibleSessions(sessions);

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
    // Navigate using replaceState to avoid full page reload
    window.history.replaceState(null, "", `/copilot?sessionId=${sessionId}`);
    // Force a re-render by updating the URL through router
    router.replace(`/copilot?sessionId=${sessionId}`);
    if (isMobile) handleCloseDrawer();
  }

  function handleNewChat() {
    resetAutoSelect();
    resetPagination();
    // Invalidate and refetch sessions list to ensure newly created sessions appear
    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey(),
    });
    window.history.replaceState(null, "", "/copilot");
    router.replace("/copilot");
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
