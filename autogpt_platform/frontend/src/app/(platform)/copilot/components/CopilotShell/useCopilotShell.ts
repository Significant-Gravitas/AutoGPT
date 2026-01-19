"use client";

import { postV2CreateSession, useGetV2GetSession, useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { okData } from "@/app/api/helpers";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
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
  const [hasAutoSelectedSession, setHasAutoSelectedSession] = useState(false);
  const hasCreatedSessionRef = useRef(false);
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

  const areAllSessionsLoaded = useMemo(() => {
    if (totalCount === null) return false;
    return accumulatedSessions.length >= totalCount && !isFetching && !isLoading;
  }, [accumulatedSessions.length, totalCount, isFetching, isLoading]);

  useEffect(() => {
    if (hasNextPage && !isFetching && !isLoading && totalCount !== null) {
      setOffset((prev) => prev + PAGE_SIZE);
    }
  }, [hasNextPage, isFetching, isLoading, totalCount]);

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
      setHasAutoSelectedSession(false);
      hasCreatedSessionRef.current = false;
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
    setHasAutoSelectedSession(false);
    hasCreatedSessionRef.current = false;
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
    if (isMobile) setIsDrawerOpen(false);
  }

  const paramSessionId = searchParams.get("sessionId");

  useEffect(() => {
    if (!areAllSessionsLoaded || hasAutoSelectedSession) return;
    
    const visibleSessions = filterVisibleSessions(accumulatedSessions);
    
    if (paramSessionId) {
      setHasAutoSelectedSession(true);
      return;
    }

    if (visibleSessions.length > 0) {
      const lastSession = visibleSessions[0];
      setHasAutoSelectedSession(true);
      router.push(`/copilot/chat?sessionId=${lastSession.id}`);
    } else if (accumulatedSessions.length === 0 && !isLoading && totalCount === 0 && !hasCreatedSessionRef.current) {
      hasCreatedSessionRef.current = true;
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
    } else if (totalCount === 0) {
      setHasAutoSelectedSession(true);
    }
  }, [areAllSessionsLoaded, accumulatedSessions, paramSessionId, hasAutoSelectedSession, router, isLoading, totalCount]);

  useEffect(() => {
    if (paramSessionId) {
      setHasAutoSelectedSession(true);
    }
  }, [paramSessionId]);

  const isReadyToShowContent = useMemo(() => {
    if (!areAllSessionsLoaded) return false;

    if (paramSessionId) {
      const sessionFound = accumulatedSessions.some((s) => s.id === paramSessionId);
      const sessionLoading = isCurrentSessionLoading;
      return sessionFound || (!sessionLoading && currentSessionData !== undefined);
    }

    return hasAutoSelectedSession;
  }, [areAllSessionsLoaded, accumulatedSessions, paramSessionId, isCurrentSessionLoading, currentSessionData, hasAutoSelectedSession]);

  return {
    isMobile,
    isDrawerOpen,
    isLoading: isLoading || !areAllSessionsLoaded,
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
    areAllSessionsLoaded,
  };
}
