"use client";

import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ChatLoader } from "@/components/contextual/Chat/components/ChatLoader/ChatLoader";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { List, Plus, X } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { Drawer } from "vaul";
import { getSessionTitle } from "./helpers";
import { useCopilotShell } from "./useCopilotShell";

interface CopilotShellProps {
  children: ReactNode;
}

export function CopilotShell({ children }: CopilotShellProps) {
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

  function renderSessionsList() {
    if (isLoading) {
      return (
        <div className="space-y-1">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="rounded-lg px-3 py-2.5">
              <Skeleton className="h-5 w-full" />
            </div>
          ))}
        </div>
      );
    }

    if (sessions.length === 0) {
      return (
        <div className="flex items-center justify-center py-8">
          <Text variant="body" className="text-zinc-500">
            No sessions found
          </Text>
        </div>
      );
    }

    return (
      <InfiniteList
        items={sessions}
        hasMore={hasNextPage}
        isFetchingMore={isFetchingNextPage}
        onEndReached={fetchNextPage}
        className="space-y-1"
        renderItem={(session) => {
          const isActive = session.id === currentSessionId;
          return (
            <button
              onClick={() => handleSelectSession(session.id)}
              className={cn(
                "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
                isActive
                  ? "bg-zinc-100"
                  : "hover:bg-zinc-50",
              )}
            >
              <Text
                variant="body"
                className={cn(
                  "font-normal",
                  isActive ? "text-zinc-600" : "text-zinc-800",
                )}
              >
                {getSessionTitle(session)}
              </Text>
            </button>
          );
        }}
      />
    );
  }

  return (
    <div
      className="flex overflow-hidden bg-zinc-50"
      style={{ height: `calc(100vh - ${NAVBAR_HEIGHT_PX}px)` }}
    >
      {!isMobile ? (
        <aside className="flex h-full w-80 flex-col border-r border-zinc-100 bg-white">
          <div className="shrink-0 px-6 py-4">
            <Text variant="h3" size="body-medium">
              Your chats
            </Text>
          </div>
          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col overflow-y-auto px-3 py-3",
              scrollbarStyles,
            )}
          >
            {renderSessionsList()}
          </div>
          <div className="shrink-0 bg-white p-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
            <Button
              variant="primary"
              size="small"
              onClick={handleNewChat}
              className="w-full"
              leftIcon={<Plus width="1rem" height="1rem" />}
            >
              New Chat
            </Button>
          </div>
        </aside>
      ) : null}

      <div className="flex min-h-0 flex-1 flex-col">
        {isMobile ? (
          <header className="flex items-center justify-between px-4 py-3">
            <Button
              variant="icon"
              size="icon"
              aria-label="Open sessions"
              onClick={handleOpenDrawer}
            >
              <List width="1.25rem" height="1.25rem" />
            </Button>
          </header>
        ) : null}
        <div className="flex min-h-0 flex-1 flex-col">
          {isReadyToShowContent ? (
            children
          ) : (
            <div className="flex flex-1 items-center justify-center">
              <div className="flex flex-col items-center gap-4">
                <ChatLoader />
                <Text variant="body" className="text-zinc-500">
                  Loading your chats...
                </Text>
              </div>
            </div>
          )}
        </div>
      </div>

      {isMobile ? (
        <Drawer.Root
          open={isDrawerOpen}
          onOpenChange={handleDrawerOpenChange}
          direction="left"
        >
          <Drawer.Portal>
            <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
            <Drawer.Content className="fixed left-0 top-0 z-[70] flex h-full w-80 flex-col border-r border-zinc-200 bg-white">
              <div className="shrink-0 border-b border-zinc-200 p-4">
                <div className="flex items-center justify-between">
                  <Drawer.Title className="text-lg font-semibold text-zinc-800">
                    Your tasks
                  </Drawer.Title>
                  <Button
                    variant="icon"
                    size="icon"
                    aria-label="Close sessions"
                    onClick={handleCloseDrawer}
                  >
                    <X width="1.25rem" height="1.25rem" />
                  </Button>
                </div>
              </div>
              <div
                className={cn(
                  "flex min-h-0 flex-1 flex-col overflow-y-auto px-3 py-3",
                  scrollbarStyles,
                )}
              >
                {renderSessionsList()}
              </div>
              <div className="shrink-0 bg-white p-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
                <Button
                  variant="primary"
                  size="small"
                  onClick={handleNewChat}
                  className="w-full"
                  leftIcon={<Plus width="1rem" height="1rem" />}
                >
                  New Chat
                </Button>
              </div>
            </Drawer.Content>
          </Drawer.Portal>
        </Drawer.Root>
      ) : null}
    </div>
  );
}
