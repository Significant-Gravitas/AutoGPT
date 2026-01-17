"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { List, X } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { Drawer } from "vaul";
import { getSessionTitle, getSessionUpdatedLabel } from "./helpers";
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
  } = useCopilotShell();

  function renderSessionsList() {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-8">
          <Text variant="body" className="text-zinc-500">
            Loading sessions...
          </Text>
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
      <div className="space-y-2">
        {sessions.map((session) => {
          const isActive = session.id === currentSessionId;
          const updatedLabel = getSessionUpdatedLabel(session);
          return (
            <button
              key={session.id}
              onClick={() => handleSelectSession(session.id)}
              className={cn(
                "w-full rounded-lg border p-3 text-left transition-colors",
                isActive
                  ? "border-indigo-500 bg-zinc-50"
                  : "border-zinc-200 bg-zinc-100/50 hover:border-zinc-300 hover:bg-zinc-50",
              )}
            >
              <div className="flex flex-col gap-1">
                <Text
                  variant="body"
                  className={cn(
                    "font-medium",
                    isActive ? "text-indigo-900" : "text-zinc-900",
                  )}
                >
                  {getSessionTitle(session)}
                </Text>
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <span>{session.id.slice(0, 8)}...</span>
                  {updatedLabel ? <span>•</span> : null}
                  <span>{updatedLabel}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-zinc-50">
      {!isMobile ? (
        <aside className="flex w-80 flex-col border-r border-zinc-200 bg-white">
          <div className="border-b border-zinc-200 px-4 py-4">
            <Text variant="h5" className="text-zinc-900">
              Chat Sessions
            </Text>
          </div>
          <div
            className={cn("flex-1 overflow-y-auto px-4 py-4", scrollbarStyles)}
          >
            {renderSessionsList()}
          </div>
        </aside>
      ) : null}

      <div className="flex min-h-screen flex-1 flex-col">
        {isMobile ? (
          <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3">
            <Button
              variant="icon"
              size="icon"
              aria-label="Open sessions"
              onClick={handleOpenDrawer}
            >
              <List width="1.25rem" height="1.25rem" />
            </Button>
            <Text variant="body-medium" className="text-zinc-700">
              Chat Sessions
            </Text>
            <div className="h-10 w-10" />
          </header>
        ) : null}
        <div className="flex-1">{children}</div>
      </div>

      {isMobile ? (
        <Drawer.Root
          open={isDrawerOpen}
          onOpenChange={handleDrawerOpenChange}
          direction="left"
        >
          <Drawer.Portal>
            <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
            <Drawer.Content
              className={cn(
                "fixed left-0 top-0 z-[70] flex h-full w-80 flex-col border-r border-zinc-200 bg-white",
                scrollbarStyles,
              )}
            >
              <div className="shrink-0 border-b border-zinc-200 p-4">
                <div className="flex items-center justify-between">
                  <Drawer.Title className="text-lg font-semibold">
                    Chat Sessions
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
              <div className="flex-1 overflow-y-auto px-4 py-4">
                {renderSessionsList()}
              </div>
            </Drawer.Content>
          </Drawer.Portal>
        </Drawer.Root>
      ) : null}
    </div>
  );
}
