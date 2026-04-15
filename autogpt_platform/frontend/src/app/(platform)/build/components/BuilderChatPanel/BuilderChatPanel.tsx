"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  ArrowCounterClockwise,
  ChatCircle,
  PaperPlaneTilt,
  SpinnerGap,
  StopCircle,
  X,
} from "@phosphor-icons/react";
import { KeyboardEvent, useEffect, useRef } from "react";
import { ToolUIPart } from "ai";
import { MessagePartRenderer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/MessagePartRenderer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { CustomNode } from "../FlowEditor/nodes/CustomNode/CustomNode";
import {
  GraphAction,
  SEED_PROMPT_PREFIX,
  extractTextFromParts,
  getActionKey,
  getNodeDisplayName,
} from "./helpers";
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
            className="pointer-events-auto flex h-[70vh] w-96 max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl"
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

function PanelHeader({
  onClose,
  undoCount,
  onUndo,
}: {
  onClose: () => void;
  undoCount: number;
  onUndo: () => void;
}) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
      <div className="flex items-center gap-2">
        <ChatCircle size={18} weight="fill" className="text-violet-600" />
        <span className="text-sm font-semibold text-slate-800">
          Chat with Builder
        </span>
      </div>
      <div className="flex items-center gap-1">
        {undoCount > 0 && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onUndo}
            aria-label="Undo last applied change"
            title="Undo last applied change"
          >
            <ArrowCounterClockwise size={16} />
          </Button>
        )}
        <Button variant="icon" size="icon" onClick={onClose} aria-label="Close">
          <X size={16} />
        </Button>
      </div>
    </div>
  );
}

interface MessageListProps {
  messages: ReturnType<typeof useBuilderChatPanel>["messages"];
  isCreatingSession: boolean;
  sessionError: boolean;
  streamError: Error | undefined;
  nodes: CustomNode[];
  parsedActions: GraphAction[];
  appliedActionKeys: Set<string>;
  onApplyAction: (action: GraphAction) => void;
  onRetry: () => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  isStreaming: boolean;
}

function MessageList({
  messages,
  isCreatingSession,
  sessionError,
  streamError,
  nodes,
  parsedActions,
  appliedActionKeys,
  onApplyAction,
  onRetry,
  messagesEndRef,
  isStreaming,
}: MessageListProps) {
  const visibleMessages = messages.filter((msg) => {
    const text = extractTextFromParts(msg.parts);
    if (msg.role === "user" && text.startsWith(SEED_PROMPT_PREFIX))
      return false;
    return (
      Boolean(text) ||
      (msg.role === "assistant" &&
        msg.parts?.some((p) => p.type === "dynamic-tool"))
    );
  });
  const lastVisibleRole = visibleMessages.at(-1)?.role;
  const showTypingIndicator =
    isStreaming && (!lastVisibleRole || lastVisibleRole === "user");

  return (
    <div
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
      className="flex-1 space-y-3 overflow-y-auto p-4"
    >
      {isCreatingSession && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <SpinnerGap size={14} className="animate-spin" />
          <span>Setting up chat session...</span>
        </div>
      )}

      {sessionError && (
        <div className="rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-600">
          <p>Failed to start chat session.</p>
          <button
            onClick={onRetry}
            className="mt-1 underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

      {streamError && (
        <div className="rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-600">
          Connection error. Please try sending your message again.
        </div>
      )}

      {visibleMessages.length === 0 && !isCreatingSession && !sessionError && (
        <div className="flex flex-col items-center gap-2 py-6 text-center text-xs text-slate-400">
          <ChatCircle size={28} weight="duotone" className="text-violet-300" />
          <p>Ask me to explain or modify your agent.</p>
          <p className="text-slate-300">
            You can say things like &ldquo;What does this agent do?&rdquo; or
            &ldquo;Add a step that formats the output.&rdquo;
          </p>
        </div>
      )}

      {visibleMessages.map((msg) => {
        const textParts = extractTextFromParts(msg.parts);

        return (
          <div
            key={msg.id}
            className={cn(
              "max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed",
              msg.role === "user"
                ? "ml-auto bg-violet-600 text-white"
                : "bg-slate-100 text-slate-800",
            )}
          >
            {msg.role === "assistant"
              ? (msg.parts ?? []).map((part, i) => {
                  // Normalize dynamic-tool parts → tool-{name} so MessagePartRenderer
                  // can route them: edit_agent/run_agent get their specific renderers,
                  // everything else falls through to GenericTool (collapsed accordion).
                  const renderedPart =
                    part.type === "dynamic-tool"
                      ? ({
                          ...part,
                          type: `tool-${(part as { toolName: string }).toolName}`,
                        } as ToolUIPart)
                      : (part as ToolUIPart);
                  return (
                    <MessagePartRenderer
                      key={`${msg.id}-${i}`}
                      part={renderedPart}
                      messageID={msg.id}
                      partIndex={i}
                    />
                  );
                })
              : textParts}
          </div>
        );
      })}

      {showTypingIndicator && <TypingIndicator />}

      {parsedActions.length > 0 && (
        <ActionList
          parsedActions={parsedActions}
          nodes={nodes}
          appliedActionKeys={appliedActionKeys}
          onApplyAction={onApplyAction}
        />
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}

function ActionList({
  parsedActions,
  nodes,
  appliedActionKeys,
  onApplyAction,
}: {
  parsedActions: GraphAction[];
  nodes: CustomNode[];
  appliedActionKeys: Set<string>;
  onApplyAction: (action: GraphAction) => void;
}) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  return (
    <div className="space-y-2 rounded-lg border border-violet-100 bg-violet-50 p-3">
      <p className="text-xs font-medium text-violet-700">Suggested changes</p>
      {parsedActions.map((action) => {
        const key = getActionKey(action);
        return (
          <ActionItem
            key={key}
            action={action}
            nodeMap={nodeMap}
            isApplied={appliedActionKeys.has(key)}
            onApply={onApplyAction}
          />
        );
      })}
    </div>
  );
}

function ActionItem({
  action,
  nodeMap,
  isApplied,
  onApply,
}: {
  action: GraphAction;
  nodeMap: Map<string, CustomNode>;
  isApplied: boolean;
  onApply: (action: GraphAction) => void;
}) {
  const label =
    action.type === "update_node_input"
      ? `Set "${getNodeDisplayName(nodeMap.get(action.nodeId), action.nodeId)}" "${action.key}" = ${JSON.stringify(action.value)}`
      : `Connect "${getNodeDisplayName(nodeMap.get(action.source), action.source)}" → "${getNodeDisplayName(nodeMap.get(action.target), action.target)}"`;

  return (
    <div className="flex items-start justify-between gap-2 rounded bg-white p-2 text-xs shadow-sm">
      <span className="leading-tight text-slate-700">{label}</span>
      {isApplied ? (
        <span className="shrink-0 rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
          Applied
        </span>
      ) : (
        <button
          onClick={() => onApply(action)}
          aria-label={`Apply: ${label}`}
          className="shrink-0 rounded bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-700 hover:bg-violet-200"
        >
          Apply
        </button>
      )}
    </div>
  );
}

interface PanelInputProps {
  value: string;
  onChange: (v: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  isDisabled: boolean;
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
}

function PanelInput({
  value,
  onChange,
  onKeyDown,
  onSend,
  onStop,
  isStreaming,
  isDisabled,
  textareaRef,
}: PanelInputProps) {
  return (
    <div className="border-t border-slate-100 p-3">
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          disabled={isDisabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your agent... (Enter to send, Shift+Enter for newline)"
          rows={2}
          maxLength={4000}
          className="flex-1 resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-200 disabled:opacity-50"
        />
        {isStreaming ? (
          <button
            onClick={onStop}
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-red-100 text-red-600 transition-colors hover:bg-red-200"
            aria-label="Stop"
          >
            <StopCircle size={18} />
          </button>
        ) : (
          <button
            onClick={onSend}
            disabled={isDisabled || !value.trim()}
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-600 text-white transition-colors hover:bg-violet-700 disabled:opacity-40"
            aria-label="Send"
          >
            <PaperPlaneTilt size={18} />
          </button>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex max-w-[85%] items-center gap-1 rounded-lg bg-slate-100 px-3 py-3">
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
    </div>
  );
}
