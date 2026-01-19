"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoader } from "./components/ChatLoader/ChatLoader";
import { useChat } from "./useChat";

export interface ChatProps {
  className?: string;
  showHeader?: boolean;
  showSessionInfo?: boolean;
  showNewChatButton?: boolean;
  onNewChat?: () => void;
  headerActions?: ReactNode;
  urlSessionId?: string | null;
  initialPrompt?: string | null;
}

export function Chat({
  className,
  showHeader = true,
  showSessionInfo = true,
  showNewChatButton = true,
  onNewChat,
  headerActions,
  urlSessionId,
  initialPrompt,
}: ChatProps) {
  const {
    messages,
    isLoading,
    isCreating,
    error,
    sessionId,
    createSession,
    clearSession,
    showLoader,
  } = useChat({ urlSessionId });

  function handleNewChat() {
    clearSession();
    onNewChat?.();
  }

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Header */}
      {showHeader && (
        <header className="shrink-0 bg-[#f8f8f9] p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {showSessionInfo && sessionId && (
                <>
                  {showNewChatButton && (
                    <Button
                      variant="outline"
                      size="small"
                      onClick={handleNewChat}
                    >
                      New Chat
                    </Button>
                  )}
                </>
              )}
              {headerActions}
            </div>
          </div>
        </header>
      )}

      {/* Main Content */}
      <main className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
        {/* Loading State - show loader when loading or creating a session (with 300ms delay) */}
        {showLoader && (isLoading || isCreating) && (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-4">
              <ChatLoader />
              <Text variant="body" className="text-zinc-500">
                Loading your chats...
              </Text>
            </div>
          </div>
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
            initialPrompt={initialPrompt}
            className="flex-1"
          />
        )}
      </main>
    </div>
  );
}
