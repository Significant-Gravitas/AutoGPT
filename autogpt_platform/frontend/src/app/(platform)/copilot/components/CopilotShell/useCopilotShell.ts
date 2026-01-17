"use client";

import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import { filterVisibleSessions } from "./helpers";

export function useCopilotShell() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const breakpoint = useBreakpoint();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const { data, isLoading } = useGetV2ListSessions(
    { limit: 100 },
    {
      query: {
        enabled: !isMobile || isDrawerOpen,
      },
    },
  );

  const sessions = useMemo(
    function getSessions() {
      if (data?.status !== 200) return [];
      return filterVisibleSessions(data.data.sessions);
    },
    [data],
  );

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
  };
}
