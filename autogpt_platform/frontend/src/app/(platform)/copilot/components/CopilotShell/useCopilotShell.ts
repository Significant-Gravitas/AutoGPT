"use client";

import {
  getGetV2GetSessionQueryKey,
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { okData } from "@/app/api/helpers";
import { useChatStore } from "@/components/contextual/Chat/chat-store";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useSearchParams } from "next/navigation";
import { useRef } from "react";
import { useCopilotStore } from "../../copilot-page-store";
import { useCopilotSessionId } from "../../useCopilotSessionId";
import { useMobileDrawer } from "./components/MobileDrawer/useMobileDrawer";
import { getCurrentSessionId } from "./helpers";
import { useShellSessionList } from "./useShellSessionList";

export function useCopilotShell() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const breakpoint = useBreakpoint();
  const { isLoggedIn } = useSupabase();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const { urlSessionId, setUrlSessionId } = useCopilotSessionId();

  const isOnHomepage = pathname === "/copilot";
  const paramSessionId = searchParams.get("sessionId");

  const {
    isDrawerOpen,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
  } = useMobileDrawer();

  const paginationEnabled = !isMobile || isDrawerOpen || !!paramSessionId;

  const currentSessionId = getCurrentSessionId(searchParams);

  const { data: currentSessionData } = useGetV2GetSession(
    currentSessionId || "",
    {
      query: {
        enabled: !!currentSessionId,
        select: okData,
      },
    },
  );

  const {
    sessions,
    isLoading,
    isSessionsFetching,
    hasNextPage,
    fetchNextPage,
    resetPagination,
    recentlyCreatedSessionsRef,
  } = useShellSessionList({
    paginationEnabled,
    currentSessionId,
    currentSessionData,
    isOnHomepage,
    paramSessionId,
  });

  const stopStream = useChatStore((s) => s.stopStream);
  const onStreamComplete = useChatStore((s) => s.onStreamComplete);
  const isStreaming = useCopilotStore((s) => s.isStreaming);
  const isCreatingSession = useCopilotStore((s) => s.isCreatingSession);
  const setIsSwitchingSession = useCopilotStore((s) => s.setIsSwitchingSession);
  const openInterruptModal = useCopilotStore((s) => s.openInterruptModal);

  const pendingActionRef = useRef<(() => void) | null>(null);

  async function stopCurrentStream() {
    if (!currentSessionId) return;

    setIsSwitchingSession(true);
    await new Promise<void>((resolve) => {
      const unsubscribe = onStreamComplete((completedId) => {
        if (completedId === currentSessionId) {
          clearTimeout(timeout);
          unsubscribe();
          resolve();
        }
      });
      const timeout = setTimeout(() => {
        unsubscribe();
        resolve();
      }, 3000);
      stopStream(currentSessionId);
    });

    queryClient.invalidateQueries({
      queryKey: getGetV2GetSessionQueryKey(currentSessionId),
    });
    setIsSwitchingSession(false);
  }

  function selectSession(sessionId: string) {
    if (sessionId === currentSessionId) return;
    if (recentlyCreatedSessionsRef.current.has(sessionId)) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
    }
    setUrlSessionId(sessionId, { shallow: false });
    if (isMobile) handleCloseDrawer();
  }

  function startNewChat() {
    resetPagination();
    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey(),
    });
    setUrlSessionId(null, { shallow: false });
    if (isMobile) handleCloseDrawer();
  }

  function handleSessionClick(sessionId: string) {
    if (sessionId === currentSessionId) return;

    if (isStreaming) {
      pendingActionRef.current = async () => {
        await stopCurrentStream();
        selectSession(sessionId);
      };
      openInterruptModal(pendingActionRef.current);
    } else {
      selectSession(sessionId);
    }
  }

  function handleNewChatClick() {
    if (isStreaming) {
      pendingActionRef.current = async () => {
        await stopCurrentStream();
        startNewChat();
      };
      openInterruptModal(pendingActionRef.current);
    } else {
      startNewChat();
    }
  }

  return {
    isMobile,
    isDrawerOpen,
    isLoggedIn,
    hasActiveSession:
      Boolean(currentSessionId) && (!isOnHomepage || Boolean(paramSessionId)),
    isLoading: isLoading || isCreatingSession,
    isCreatingSession,
    sessions,
    currentSessionId: urlSessionId,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChatClick,
    handleSessionClick,
    hasNextPage,
    isFetchingNextPage: isSessionsFetching,
    fetchNextPage,
  };
}
