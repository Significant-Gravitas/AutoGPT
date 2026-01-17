"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoadingState } from "./components/ChatLoadingState/ChatLoadingState";
import { useChat } from "./useChat";

export interface ChatProps {
  className?: string;
  headerTitle?: ReactNode;
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
  headerTitle = "AutoGPT Copilot",
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
  } = useChat({ urlSessionId });

  function handleNewChat() {
    clearSession();
    onNewChat?.();
  }

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Header */}
      {showHeader && (
        <header className="shrink-0 border-t border-zinc-200 bg-white p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {typeof headerTitle === "string" ? (
                <Text variant="h2" className="text-lg font-semibold">
                  {headerTitle}
                </Text>
              ) : (
                headerTitle
              )}
            </div>
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
            initialPrompt={initialPrompt}
            className="flex-1"
          />
        )}
      </main>
    </div>
  );
}
