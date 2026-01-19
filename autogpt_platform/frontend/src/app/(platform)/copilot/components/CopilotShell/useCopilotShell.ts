"use client";

import { useGetV2GetSession, useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { okData } from "@/app/api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { filterVisibleSessions } from "./helpers";

function convertSessionDetailToSummary(
  session: SessionDetailResponse,
): SessionSummaryResponse {
  return {
    id: session.id,
    created_at: session.created_at,
    updated_at: session.updated_at,
    title: undefined,
  };
}

export function useCopilotShell() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const breakpoint = useBreakpoint();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const [offset, setOffset] = useState(0);
  const [accumulatedSessions, setAccumulatedSessions] = useState<SessionSummaryResponse[]>([]);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const PAGE_SIZE = 50;

  const { data, isLoading, isFetching } = useGetV2ListSessions(
    { limit: PAGE_SIZE, offset },
    {
      query: {
        enabled: (!isMobile || isDrawerOpen) && offset >= 0,
      },
    },
  );

  useEffect(() => {
    const responseData = okData(data);
    if (responseData) {
      const newSessions = responseData.sessions;
      const total = responseData.total;
      setTotalCount(total);
      
      if (offset === 0) {
        setAccumulatedSessions(newSessions);
      } else {
        setAccumulatedSessions((prev) => [...prev, ...newSessions]);
      }
    }
  }, [data, offset]);

  const hasNextPage = useMemo(() => {
    if (totalCount === null) return false;
    return accumulatedSessions.length < totalCount;
  }, [accumulatedSessions.length, totalCount]);

  const fetchNextPage = () => {
    if (hasNextPage && !isFetching) {
      setOffset((prev) => prev + PAGE_SIZE);
    }
  };

  // Reset when query becomes disabled (mobile with drawer closed)
  useEffect(() => {
    const isQueryEnabled = !isMobile || isDrawerOpen;
    if (!isQueryEnabled) {
      setOffset(0);
      setAccumulatedSessions([]);
      setTotalCount(null);
    }
  }, [isMobile, isDrawerOpen]);

  const currentSessionId = useMemo(
    function getCurrentSessionId() {
      const paramSessionId = searchParams.get("sessionId");
      if (paramSessionId) return paramSessionId;
      const storedSessionId = storage.get(Key.CHAT_SESSION_ID);
      if (storedSessionId) return storedSessionId;
      return null;
    },
    [searchParams],
  );

  const { data: currentSessionData, isLoading: isCurrentSessionLoading } = useGetV2GetSession(
    currentSessionId || "",
    {
      query: {
        enabled: !!currentSessionId && (!isMobile || isDrawerOpen),
        select: okData,
      },
    },
  );

  const sessions = useMemo(
    function getSessions() {
      const filteredSessions: SessionSummaryResponse[] = [];
      
      if (accumulatedSessions.length > 0) {
        const visibleSessions = filterVisibleSessions(accumulatedSessions);
        
        if (currentSessionId) {
          const currentInAll = accumulatedSessions.find((s) => s.id === currentSessionId);
          if (currentInAll) {
            const isInVisible = visibleSessions.some((s) => s.id === currentSessionId);
            if (!isInVisible) {
              filteredSessions.push(currentInAll);
            }
          }
        }
        
        filteredSessions.push(...visibleSessions);
      }

      if (currentSessionId && currentSessionData) {
        const isCurrentInList = filteredSessions.some(
          (s) => s.id === currentSessionId,
        );
        if (!isCurrentInList) {
          const summarySession = convertSessionDetailToSummary(currentSessionData);
          // Add new session at the beginning to match API order (most recent first)
          filteredSessions.unshift(summarySession);
        }
      }

      return filteredSessions;
    },
    [accumulatedSessions, currentSessionId, currentSessionData],
  );

  function handleSelectSession(sessionId: string) {
    router.push(`/copilot/chat?sessionId=${sessionId}`);
    if (isMobile) setIsDrawerOpen(false);
  }

  function handleOpenDrawer() {
    setIsDrawerOpen(true);
  }

  function handleCloseDrawer() {
    setIsDrawerOpen(false);
  }

  function handleDrawerOpenChange(open: boolean) {
    setIsDrawerOpen(open);
  }

  function handleNewChat() {
    storage.clean(Key.CHAT_SESSION_ID);
    router.push("/copilot");
    if (isMobile) setIsDrawerOpen(false);
  }

  // Determine if we're ready to show the main content
  // We need to wait for:
  // 1. Sessions to load (at least first page)
  // 2. If there's a sessionId query param, wait for that session to be found/loaded
  const isReadyToShowContent = useMemo(() => {
    // If still loading initial sessions, not ready
    if (isLoading && accumulatedSessions.length === 0) {
      return false;
    }

    // If there's a sessionId query param, wait for it to be found/loaded
    const paramSessionId = searchParams.get("sessionId");
    if (paramSessionId) {
      // Check if session is in accumulated sessions or if we're still loading it
      const sessionFound = accumulatedSessions.some((s) => s.id === paramSessionId);
      const sessionLoading = isCurrentSessionLoading;
      
      // Ready if session is found OR if we've finished loading it (even if not in list)
      return sessionFound || (!sessionLoading && currentSessionData !== undefined);
    }

    // No sessionId param, ready once sessions have loaded
    return !isLoading || accumulatedSessions.length > 0;
  }, [isLoading, accumulatedSessions, searchParams, isCurrentSessionLoading, currentSessionData]);

  return {
    isMobile,
    isDrawerOpen,
    isLoading,
    sessions,
    currentSessionId,
    handleSelectSession,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChat,
    hasNextPage: hasNextPage ?? false,
    isFetchingNextPage: isFetching,
    fetchNextPage,
    isReadyToShowContent,
  };
}
