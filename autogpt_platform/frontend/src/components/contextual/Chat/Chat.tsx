"use client";

import { cn } from "@/lib/utils";
import React from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoadingState } from "./components/ChatLoadingState/ChatLoadingState";
import { useChat } from "./useChat";

export interface ChatProps {
  className?: string;
  headerTitle?: React.ReactNode;
  showHeader?: boolean;
  showSessionInfo?: boolean;
  showNewChatButton?: boolean;
  onNewChat?: () => void;
  headerActions?: React.ReactNode;
}

export function Chat({
  className,
  headerTitle = "Chat",
  showHeader = true,
  showSessionInfo = true,
  showNewChatButton = true,
  onNewChat,
  headerActions,
}: ChatProps) {
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

  const handleNewChat = () => {
    clearSession();
    onNewChat?.();
  };

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Header */}
      {showHeader && (
        <header className="shrink-0 border-b border-zinc-200 bg-white p-4">
          <div className="flex items-center justify-between">
            {typeof headerTitle === "string" ? (
              <h2 className="text-xl font-semibold">{headerTitle}</h2>
            ) : (
              headerTitle
            )}
            <div className="flex items-center gap-4">
              {showSessionInfo && sessionId && (
                <>
                  <span className="text-sm text-zinc-600">
                    Session: {sessionId.slice(0, 8)}...
                  </span>
                  {showNewChatButton && (
                    <button
                      onClick={handleNewChat}
                      className="text-sm text-zinc-600 hover:text-zinc-900"
                    >
                      New Chat
                    </button>
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
            onRefreshSession={refreshSession}
            className="flex-1"
          />
        )}
      </main>
    </div>
  );
}
