"use client";

import { useChatPage } from "./useChatPage";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoadingState } from "./components/ChatLoadingState/ChatLoadingState";
import { useGetFlag, Flag } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function ChatPage() {
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const router = useRouter();
  const {
    messages,
    isLoading,
    isCreating,
    error,
    sessionId,
    createSession,
    clearSession,
    refreshSession,
  } = useChatPage();

  useEffect(() => {
    if (isChatEnabled === false) {
      router.push("/404");
    }
  }, [isChatEnabled, router]);

  if (isChatEnabled === null || isChatEnabled === false) {
    return null;
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="container mx-auto flex items-center justify-between">
          <h1 className="text-xl font-semibold">Chat</h1>
          {sessionId && (
            <div className="flex items-center gap-4">
              <span className="text-sm text-zinc-600 dark:text-zinc-400">
                Session: {sessionId.slice(0, 8)}...
              </span>
              <button
                onClick={clearSession}
                className="text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
              >
                New Chat
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto flex flex-1 flex-col overflow-hidden">
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
