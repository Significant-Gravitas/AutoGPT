"use client";

import { cn } from "@/lib/utils";
import { ChatCircle, X } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { MessageList } from "./components/MessageList";
import { PanelHeader } from "./components/PanelHeader";
import { PanelInput } from "./components/PanelInput";
import { useBuilderChatPanel } from "./useBuilderChatPanel";

interface Props {
  className?: string;
  isGraphLoaded?: boolean;
  onGraphEdited?: () => void;
}

export function BuilderChatPanel({
  className,
  isGraphLoaded,
  onGraphEdited,
}: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const {
    isOpen,
    handleToggle,
    retrySession,
    messages,
    stop,
    error,
    isCreatingSession,
    sessionError,
    nodes,
    parsedActions,
    appliedActionKeys,
    handleApplyAction,
    undoStack,
    handleUndoLastAction,
    inputValue,
    setInputValue,
    handleSend,
    sendRawMessage,
    handleKeyDown,
    isStreaming,
    canSend,
  } = useBuilderChatPanel({ isGraphLoaded, onGraphEdited, panelRef });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  // Move focus to the textarea when the panel opens so keyboard users can type immediately.
  useEffect(() => {
    if (isOpen) {
      textareaRef.current?.focus();
    }
  }, [isOpen]);

  return (
    <div
      className={cn(
        "pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2",
        className,
      )}
    >
      {isOpen && (
        <CopilotChatActionsProvider onSend={sendRawMessage}>
          <div
            ref={panelRef}
            role="complementary"
            aria-label="Builder chat panel"
            // max-h-[70vh] instead of h-[70vh] so the panel shrinks with the
            // viewport on small screens and does not overlap the builder toolbar.
            className="pointer-events-auto flex max-h-[70vh] min-h-[320px] w-96 max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl sm:max-h-[75vh]"
          >
            <PanelHeader
              onClose={handleToggle}
              undoCount={undoStack.length}
              onUndo={handleUndoLastAction}
            />

            <MessageList
              messages={messages}
              isCreatingSession={isCreatingSession}
              sessionError={sessionError}
              streamError={error}
              nodes={nodes}
              parsedActions={parsedActions}
              appliedActionKeys={appliedActionKeys}
              onApplyAction={handleApplyAction}
              onRetry={retrySession}
              messagesEndRef={messagesEndRef}
              isStreaming={isStreaming}
            />

            <PanelInput
              value={inputValue}
              onChange={setInputValue}
              onKeyDown={handleKeyDown}
              onSend={handleSend}
              onStop={stop}
              isStreaming={isStreaming}
              isDisabled={!canSend}
              textareaRef={textareaRef}
            />
          </div>
        </CopilotChatActionsProvider>
      )}

      <button
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
