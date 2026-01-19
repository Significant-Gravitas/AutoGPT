"use client";

import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import type { ReactNode } from "react";
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

  return (
    <div
      className="flex overflow-hidden bg-zinc-50"
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
          onNewChat={handleNewChat}
        />
      )}

      <div className="flex min-h-0 flex-1 flex-col">
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
          onNewChat={handleNewChat}
          onClose={handleCloseDrawer}
          onOpenChange={handleDrawerOpenChange}
        />
      )}
    </div>
  );
}
