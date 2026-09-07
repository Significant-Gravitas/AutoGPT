"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { cn } from "@/lib/utils";
import { useRef } from "react";
import { PanelHeader } from "./components/PanelHeader";
import { useBuilderChatPanel } from "./useBuilderChatPanel";
import { BubbleChatIcon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
    bindError,
    bootstrapError,
    retryBind,
    retryBootstrap,
  } = useBuilderChatPanel({ panelRef });

  const isStreaming = status === "streaming" || status === "submitted";
  const activeError = bindError ?? bootstrapError ?? null;
  const activeRetry = bindError
    ? retryBind
    : bootstrapError
      ? retryBootstrap
      : null;
  const activeErrorTitle = bindError
    ? "Could not start the builder chat"
    : "Could not create a blank agent";

  return (
    <div
      className={cn(
        "pointer-events-none fixed z-50 flex flex-col",
        isOpen
          ? "inset-0 items-stretch gap-0 sm:bottom-4 sm:left-auto sm:right-4 sm:top-auto sm:items-end sm:gap-2"
          : "bottom-[max(1rem,env(safe-area-inset-bottom))] right-4 items-end gap-2",
        className,
      )}
    >
      {isOpen && (
        <CopilotChatActionsProvider onSend={onSend} chatSurface="builder">
          <div
            ref={panelRef}
            role="complementary"
            aria-label="Builder chat panel"
            className="pointer-events-auto flex h-dvh max-h-none w-full max-w-none flex-col overflow-hidden bg-white sm:h-[75vh] sm:max-h-[calc(100dvh-6rem)] sm:w-[26rem] sm:max-w-[calc(100vw-2rem)] sm:rounded-xl sm:border sm:border-slate-200 sm:shadow-2xl"
          >
            <PanelHeader
              onClose={handleToggle}
              canRevert={revertTargetVersion != null}
              revertTargetVersion={revertTargetVersion}
              onRevert={handleRevert}
            />

            <div className="flex h-0 min-h-0 flex-1 flex-col">
              {activeError && activeRetry ? (
                <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-6 text-center text-sm text-slate-600">
                  <p className="font-medium text-slate-800">
                    {activeErrorTitle}
                  </p>
                  <p className="text-slate-500">
                    Something went wrong. Retry to try again.
                  </p>
                  <button
                    type="button"
                    onClick={activeRetry}
                    className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                  >
                    Retry
                  </button>
                </div>
              ) : isBootstrapping ? (
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
                  <div className="relative shrink-0 border-t border-slate-100 bg-white px-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] pt-2">
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
          "pointer-events-auto h-12 w-12 items-center justify-center rounded-full shadow-lg transition-colors",
          isOpen ? "hidden sm:flex" : "flex",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2",
          isOpen
            ? "bg-slate-800 text-white hover:bg-slate-700"
            : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
        )}
      >
        {isOpen ? (
          <Icon icon={Cancel01Icon} size={20} />
        ) : (
          <Icon icon={BubbleChatIcon} size={22} />
        )}
      </button>
    </div>
  );
}
