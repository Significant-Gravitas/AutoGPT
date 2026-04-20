"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { cn } from "@/lib/utils";
import { ChatCircle, X } from "@phosphor-icons/react";
import { useRef } from "react";
import { PanelHeader } from "./components/PanelHeader";
import { useBuilderChatPanel } from "./useBuilderChatPanel";

interface Props {
  className?: string;
}

export function BuilderChatPanel({ className }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const {
    isOpen,
    handleToggle,
    sessionId,
    messages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
    isBootstrapping,
    revertTargetVersion,
    handleRevert,
  } = useBuilderChatPanel({ panelRef });

  const isStreaming = status === "streaming" || status === "submitted";

  return (
    <div
      className={cn(
        "pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2",
        className,
      )}
    >
      {isOpen && (
        <CopilotChatActionsProvider onSend={onSend} chatSurface="builder">
          <div
            ref={panelRef}
            role="complementary"
            aria-label="Builder chat panel"
            className="pointer-events-auto flex h-[70vh] max-h-[calc(100vh-6rem)] w-[26rem] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl sm:h-[75vh]"
          >
            <PanelHeader
              onClose={handleToggle}
              canRevert={revertTargetVersion != null}
              revertTargetVersion={revertTargetVersion}
              onRevert={handleRevert}
            />

            <div className="flex h-0 min-h-0 flex-1 flex-col">
              {isBootstrapping ? (
                <div className="flex flex-1 items-center justify-center px-4 py-6 text-sm text-slate-500">
                  Preparing builder chat…
                </div>
              ) : sessionId ? (
                <>
                  <div className="flex min-h-0 flex-1 flex-col">
                    <ChatMessagesContainer
                      messages={messages}
                      status={status}
                      error={error}
                      isLoading={false}
                      sessionID={sessionId}
                      queuedMessages={queuedMessages}
                    />
                  </div>
                  <div className="relative shrink-0 border-t border-slate-100 bg-white px-3 pb-2 pt-2">
                    <ChatInput
                      inputId="builder-chat-input"
                      onSend={onSend}
                      disabled={false}
                      isStreaming={isStreaming}
                      onStop={stop}
                      onEnqueue={onSend}
                      placeholder="Ask the builder to edit or run this agent…"
                      hasSession={true}
                    />
                  </div>
                </>
              ) : (
                <div className="flex flex-1 items-center justify-center px-4 py-6 text-sm text-slate-500">
                  Open an agent to start chatting with the builder.
                </div>
              )}
            </div>
          </div>
        </CopilotChatActionsProvider>
      )}

      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-label={isOpen ? "Close chat" : "Chat with builder"}
        className={cn(
          "pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full shadow-lg transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2",
          isOpen
            ? "bg-slate-800 text-white hover:bg-slate-700"
            : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
        )}
      >
        {isOpen ? <X size={20} /> : <ChatCircle size={22} weight="fill" />}
      </button>
    </div>
  );
}
