"use client";

import { useChatPage } from "./useChatPage";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoadingState } from "./components/ChatLoadingState/ChatLoadingState";

function ChatPage() {
  const {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    sessionId,
    createSession,
    clearSession,
    refreshSession,
  } = useChatPage();

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
        {/* Loading State */}
        {(isLoading || isCreating) && (
          <ChatLoadingState
            message={isCreating ? "Creating session..." : "Loading..."}
          />
        )}

        {/* Error State */}
        {error && !isLoading && (
          <ChatErrorState error={error} onRetry={createSession} />
        )}

        {/* Session Content */}
        {session && !isLoading && !error && (
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

export default ChatPage;
