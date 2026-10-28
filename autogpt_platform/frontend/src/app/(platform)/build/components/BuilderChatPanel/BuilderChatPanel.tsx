"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { cn } from "@/lib/utils";
import { useRef } from "react";
import { NoModelNotice } from "./components/NoModelNotice";
import { PanelHeader } from "./components/PanelHeader";
import { useBuilderChatPanel } from "./useBuilderChatPanel";
import { BubbleChatIcon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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
    hasNoModel,
    revertTargetVersion,
    handleRevert,
    bindError,
    bootstrapError,
    retryBind,
    retryBootstrap,
    transportCheckError,
    retryTransportCheck,
  } = useBuilderChatPanel({ panelRef });

  const isStreaming = status === "streaming" || status === "submitted";
  const activeError = bindError ?? bootstrapError ?? transportCheckError;
  const activeRetry = bindError
    ? retryBind
    : bootstrapError
      ? retryBootstrap
      : transportCheckError
        ? retryTransportCheck
        : null;
  const activeErrorTitle = bindError
    ? "Could not start the builder chat"
    : bootstrapError
      ? "Could not create a blank workflow"
      : "Could not check the AI connection";

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
            // TODO(tokens): no fitting role for border-slate-200 #CFD4DB; #CFD4DB is node-border (the builder node ring), node-selected (the selected builder node ring)
            className="pointer-events-auto flex h-[70vh] max-h-[calc(100vh-6rem)] w-[26rem] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-2xl sm:h-[75vh]"
          >
            <PanelHeader
              onClose={handleToggle}
              canRevert={revertTargetVersion != null}
              revertTargetVersion={revertTargetVersion}
              onRevert={handleRevert}
            />

            <div className="flex h-0 min-h-0 flex-1 flex-col">
              {hasNoModel ? (
                <NoModelNotice />
              ) : activeError && activeRetry ? (
                <Text
                  variant="body-tight"
                  as="div"
                  // TODO(tokens): no exact role: text-slate-600 #5B6A7E
                  className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-6 text-center text-slate-600"
                >
                  <Text
                    variant="body-medium-tight"
                    // TODO(tokens): no exact role: text-slate-800 #475263
                    className="text-slate-800"
                  >
                    {activeErrorTitle}
                  </Text>
                  <Text
                    variant="body-tight"
                    // TODO(tokens): no exact role: text-slate-500 #64748B
                    className="text-slate-500"
                  >
                    Something went wrong. Retry to try again.
                  </Text>
                  <button
                    type="button"
                    onClick={activeRetry}
                    // TODO(tokens): no exact role: border-slate-300 #B8BFCA, text-slate-700 #475263, hover:bg-slate-50 #F7F8F9, focus-visible:ring-violet-400 #A78BFA
                    className="rounded-md border border-slate-300 bg-surface px-3 py-1.5 text-slate-700 shadow-sm hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                  >
                    <Text variant="body-medium-tight" as="span" tone="inherit">
                      Retry
                    </Text>
                  </button>
                </Text>
              ) : isBootstrapping ? (
                <Text
                  variant="body-tight"
                  as="div"
                  // TODO(tokens): no exact role: text-slate-500 #64748B
                  className="flex flex-1 items-center justify-center px-4 py-6 text-slate-500"
                >
                  Preparing builder chat…
                </Text>
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
                  <div
                    // TODO(tokens): no exact role: border-slate-100 #EFF1F4
                    className="relative shrink-0 border-t border-slate-100 bg-surface px-3 pb-2 pt-2"
                  >
                    <ChatInput
                      inputId="builder-chat-input"
                      onSend={onSend}
                      disabled={false}
                      isStreaming={isStreaming}
                      isTurnStarting={status === "submitted"}
                      onStop={stop}
                      onEnqueue={onSend}
                      placeholder="Ask the builder to edit or run this workflow…"
                      hasSession={true}
                    />
                  </div>
                </>
              ) : (
                <Text
                  variant="body-tight"
                  as="div"
                  // TODO(tokens): no exact role: text-slate-500 #64748B
                  className="flex flex-1 items-center justify-center px-4 py-6 text-slate-500"
                >
                  Open a workflow to start chatting with the builder.
                </Text>
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
          // TODO(tokens): no exact role: focus-visible:ring-violet-400 #A78BFA
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2",
          isOpen
            ? // TODO(tokens): no exact role: bg-slate-800 #475263, hover:bg-slate-700 #475263
              "bg-slate-800 text-action-foreground hover:bg-slate-700"
            : // TODO(tokens): no exact role: text-slate-700 #475263, hover:bg-slate-50 #F7F8F9; no fitting role for border-slate-200 #CFD4DB; #CFD4DB is node-border (the builder node ring), node-selected (the selected builder node ring)
              "border border-slate-200 bg-surface text-slate-700 hover:bg-slate-50",
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
