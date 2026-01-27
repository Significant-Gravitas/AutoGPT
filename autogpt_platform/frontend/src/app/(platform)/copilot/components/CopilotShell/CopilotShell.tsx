"use client";

import { ChatLoader } from "@/components/contextual/Chat/components/ChatLoader/ChatLoader";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import type { ReactNode } from "react";
import { useCallback, useEffect } from "react";
import { useCopilotStore } from "../../copilot-page-store";
import { DesktopSidebar } from "./components/DesktopSidebar/DesktopSidebar";
import { LoadingState } from "./components/LoadingState/LoadingState";
import { MobileDrawer } from "./components/MobileDrawer/MobileDrawer";
import { MobileHeader } from "./components/MobileHeader/MobileHeader";
import { useCopilotShell } from "./useCopilotShell";

interface Props {
  children: ReactNode;
}

export function CopilotShell({ children }: Props) {
  const {
    isMobile,
    isDrawerOpen,
    isLoading,
    isLoggedIn,
    hasActiveSession,
    sessions,
    currentSessionId,
    handleSelectSession,
    performSelectSession,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChat,
    performNewChat,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    isReadyToShowContent,
  } = useCopilotShell();

  const setNewChatHandler = useCopilotStore((s) => s.setNewChatHandler);
  const setNewChatWithInterruptHandler = useCopilotStore(
    (s) => s.setNewChatWithInterruptHandler,
  );
  const setSelectSessionHandler = useCopilotStore(
    (s) => s.setSelectSessionHandler,
  );
  const setSelectSessionWithInterruptHandler = useCopilotStore(
    (s) => s.setSelectSessionWithInterruptHandler,
  );
  const requestNewChat = useCopilotStore((s) => s.requestNewChat);
  const requestSelectSession = useCopilotStore((s) => s.requestSelectSession);

  const stableHandleNewChat = useCallback(handleNewChat, [handleNewChat]);
  const stablePerformNewChat = useCallback(performNewChat, [performNewChat]);

  useEffect(
    function registerNewChatHandlers() {
      setNewChatHandler(stableHandleNewChat);
      setNewChatWithInterruptHandler(stablePerformNewChat);
      return function cleanup() {
        setNewChatHandler(null);
        setNewChatWithInterruptHandler(null);
      };
    },
    [
      stableHandleNewChat,
      stablePerformNewChat,
      setNewChatHandler,
      setNewChatWithInterruptHandler,
    ],
  );

  const stableHandleSelectSession = useCallback(handleSelectSession, [
    handleSelectSession,
  ]);

  const stablePerformSelectSession = useCallback(performSelectSession, [
    performSelectSession,
  ]);

  useEffect(
    function registerSelectSessionHandlers() {
      setSelectSessionHandler(stableHandleSelectSession);
      setSelectSessionWithInterruptHandler(stablePerformSelectSession);
      return function cleanup() {
        setSelectSessionHandler(null);
        setSelectSessionWithInterruptHandler(null);
      };
    },
    [
      stableHandleSelectSession,
      stablePerformSelectSession,
      setSelectSessionHandler,
      setSelectSessionWithInterruptHandler,
    ],
  );

  function handleNewChatClick() {
    requestNewChat();
  }

  function handleSessionClick(sessionId: string) {
    requestSelectSession(sessionId);
  }

  if (!isLoggedIn) {
    return (
      <div className="flex h-full items-center justify-center">
        <ChatLoader />
      </div>
    );
  }

  return (
    <div
      className="flex overflow-hidden bg-[#EFEFF0]"
      style={{ height: `calc(100vh - ${NAVBAR_HEIGHT_PX}px)` }}
    >
      {!isMobile && (
        <DesktopSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          isLoading={isLoading}
          hasNextPage={hasNextPage}
          isFetchingNextPage={isFetchingNextPage}
          onSelectSession={handleSessionClick}
          onFetchNextPage={fetchNextPage}
          onNewChat={handleNewChatClick}
          hasActiveSession={Boolean(hasActiveSession)}
        />
      )}

      <div className="relative flex min-h-0 flex-1 flex-col">
        {isMobile && <MobileHeader onOpenDrawer={handleOpenDrawer} />}
        <div className="flex min-h-0 flex-1 flex-col">
          {isReadyToShowContent ? children : <LoadingState />}
        </div>
      </div>

      {isMobile && (
        <MobileDrawer
          isOpen={isDrawerOpen}
          sessions={sessions}
          currentSessionId={currentSessionId}
          isLoading={isLoading}
          hasNextPage={hasNextPage}
          isFetchingNextPage={isFetchingNextPage}
          onSelectSession={handleSessionClick}
          onFetchNextPage={fetchNextPage}
          onNewChat={handleNewChatClick}
          onClose={handleCloseDrawer}
          onOpenChange={handleDrawerOpenChange}
          hasActiveSession={Boolean(hasActiveSession)}
        />
      )}
    </div>
  );
}
