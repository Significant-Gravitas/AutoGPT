"use client";

import { ChatLoader } from "@/components/contextual/Chat/components/ChatLoader/ChatLoader";
import { Text } from "@/components/atoms/Text/Text";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import type { ReactNode } from "react";
import { DesktopSidebar } from "./components/DesktopSidebar/DesktopSidebar";
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
    isCreatingSession,
    isLoggedIn,
    hasActiveSession,
    sessions,
    currentSessionId,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleNewChatClick,
    handleSessionClick,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useCopilotShell();

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
          {isCreatingSession ? (
            <div className="flex h-full flex-1 flex-col items-center justify-center bg-[#f8f8f9]">
              <div className="flex flex-col items-center gap-4">
                <ChatLoader />
                <Text variant="body" className="text-zinc-500">
                  Creating your chat...
                </Text>
              </div>
            </div>
          ) : (
            children
          )}
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
