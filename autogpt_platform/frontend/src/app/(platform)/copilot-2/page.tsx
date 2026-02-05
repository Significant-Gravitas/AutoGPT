"use client";

import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { CopyIcon, CheckIcon } from "@phosphor-icons/react";
import { useCopilotPage } from "./useCopilotPage";

export default function Page() {
  const {
    copied,
    sessionId,
    messages,
    status,
    error,
    isCreatingSession,
    handleCopySessionId,
    createSession,
    onSend,
  } = useCopilotPage();

  return (
    <SidebarProvider
      defaultOpen={false}
      className="h-[calc(100vh-72px)] min-h-0"
    >
      <ChatSidebar />
      <SidebarInset className="relative flex h-[calc(100vh-80px)] flex-col overflow-hidden ring-1 ring-zinc-300">
        {sessionId && (
          <div className="absolute flex items-center px-4 py-4">
            <div className="flex items-center gap-2 rounded-3xl border border-neutral-400 bg-neutral-100 px-3 py-1.5 text-sm text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
              <span className="text-xs">{sessionId.slice(0, 8)}...</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={handleCopySessionId}
              >
                {copied ? (
                  <CheckIcon className="h-3.5 w-3.5 text-green-500" />
                ) : (
                  <CopyIcon className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>
          </div>
        )}
        <div className="flex-1 overflow-hidden">
          <ChatContainer
            messages={messages}
            status={status}
            error={error}
            sessionId={sessionId}
            isCreatingSession={isCreatingSession}
            onCreateSession={createSession}
            onSend={onSend}
          />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
