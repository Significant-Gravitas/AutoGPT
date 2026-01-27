"use client";

import { ChatLoader } from "@/components/contextual/Chat/components/ChatLoader/ChatLoader";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import type { ReactNode } from "react";
import { useEffect } from "react";
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
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChat,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    isReadyToShowContent,
  } = useCopilotShell();

  const setNewChatHandler = useCopilotStore((s) => s.setNewChatHandler);
  const requestNewChat = useCopilotStore((s) => s.requestNewChat);

  useEffect(
    function registerNewChatHandler() {
      setNewChatHandler(handleNewChat);
      return function cleanup() {
        setNewChatHandler(null);
      };
    },
    [setNewChatHandler, handleNewChat],
  );

  function handleNewChatClick() {
    requestNewChat();
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
          onSelectSession={handleSelectSession}
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
          onSelectSession={handleSelectSession}
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
