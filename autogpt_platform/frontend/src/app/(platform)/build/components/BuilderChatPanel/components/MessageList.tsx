import { cn } from "@/lib/utils";
import { ChatCircle, SpinnerGap } from "@phosphor-icons/react";
import { ToolUIPart } from "ai";
import { MessagePartRenderer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/MessagePartRenderer";
import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";
import {
  GraphAction,
  SEED_PROMPT_PREFIX,
  extractTextFromParts,
} from "../helpers";
import { useBuilderChatPanel } from "../useBuilderChatPanel";
import { ActionList } from "./ActionList";
import { TypingIndicator } from "./TypingIndicator";

/**
 * Runtime guard: does `part` look like an AI SDK dynamic-tool part?
 *
 * Dynamic-tool parts have a string `toolName`, which `MessagePartRenderer`
 * needs to route to the correct tool-specific renderer.
 */
function isDynamicToolPart(
  part: unknown,
): part is { type: "dynamic-tool"; toolName: string } {
  if (typeof part !== "object" || part === null) return false;
  const p = part as { type?: unknown; toolName?: unknown };
  return p.type === "dynamic-tool" && typeof p.toolName === "string";
}

/**
 * Normalize a message part for the copilot `MessagePartRenderer`.
 *
 * The AI SDK emits `dynamic-tool` parts with a separate `toolName`, while
 * `MessagePartRenderer` dispatches on `type === "tool-<name>"`. Rewriting the
 * type here lets `edit_agent`/`run_agent` get their specific renderers and
 * everything else fall through to `GenericTool` (collapsed accordion).
 *
 * Exported for direct unit testing — the runtime type guard and cast live
 * here so they can be covered without mounting the full MessageList.
 */
export function normalizePartForRenderer(part: unknown): ToolUIPart {
  if (isDynamicToolPart(part)) {
    // MessagePartRenderer only reads `type`, `toolCallId`, `state`, and
    // `output` from the part, so preserving the extra `toolName` key is safe
    // — the structural mismatch with the narrower `ToolUIPart` union is
    // intentional and only surfaces at the cast boundary.
    return {
      ...part,
      type: `tool-${part.toolName}`,
    } as unknown as ToolUIPart;
  }
  return part as ToolUIPart;
}

interface Props {
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

export function MessageList({
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
}: Props) {
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
              ? (msg.parts ?? []).map((part, i) => (
                  <MessagePartRenderer
                    key={`${msg.id}-${i}`}
                    part={normalizePartForRenderer(part)}
                    messageID={msg.id}
                    partIndex={i}
                  />
                ))
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
