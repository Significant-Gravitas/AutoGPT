"use client";

import { Button } from "@/components/__legacy__/ui/button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { X } from "@phosphor-icons/react";
import { useEffect } from "react";
import { Drawer } from "vaul";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoadingState } from "./components/ChatLoadingState/ChatLoadingState";
import { useChat } from "./useChat";
import { useChatDrawer } from "./useChatDrawer";

export function ChatDrawer() {
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const { isOpen, close } = useChatDrawer();
  const {
    messages,
    isLoading,
    isCreating,
    error,
    sessionId,
    createSession,
    clearSession,
    refreshSession,
  } = useChat();

  useEffect(() => {
    if (isChatEnabled === false && isOpen) {
      close();
    }
  }, [isChatEnabled, isOpen, close]);

  if (isChatEnabled === null || isChatEnabled === false) {
    return null;
  }

  return (
    <Drawer.Root
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          close();
        }
      }}
      direction="right"
      modal={false}
    >
      <Drawer.Portal>
        <Drawer.Content
          className={cn(
            "fixed right-0 top-0 z-50 flex h-full w-1/2 flex-col border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900",
            scrollbarStyles,
          )}
        >
          {/* Header */}
          <header className="shrink-0 border-b border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center justify-between">
              <Drawer.Title className="text-xl font-semibold">
                Chat
              </Drawer.Title>
              <div className="flex items-center gap-4">
                {sessionId && (
                  <>
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">
                      Session: {sessionId.slice(0, 8)}...
                    </span>
                    <button
                      onClick={clearSession}
                      className="text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                    >
                      New Chat
                    </button>
                  </>
                )}
                <Button
                  variant="link"
                  aria-label="Close"
                  onClick={close}
                  className="!focus-visible:ring-0 p-0"
                >
                  <X width="1.5rem" />
                </Button>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
            {/* Loading State - show when explicitly loading/creating OR when we don't have a session yet and no error */}
            {(isLoading || isCreating || (!sessionId && !error)) && (
              <ChatLoadingState
                message={isCreating ? "Creating session..." : "Loading..."}
              />
            )}

            {/* Error State */}
            {error && !isLoading && (
              <ChatErrorState error={error} onRetry={createSession} />
            )}

            {/* Session Content */}
            {sessionId && !isLoading && !error && (
              <ChatContainer
                sessionId={sessionId}
                initialMessages={messages}
                onRefreshSession={refreshSession}
                className="flex-1"
              />
            )}
          </main>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
