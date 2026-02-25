import type { UIMessage, UIDataTypes, UITools } from "ai";

interface SessionChatMessage {
  role: string;
  content: string | null;
  tool_call_id: string | null;
  tool_calls: unknown[] | null;
}

function coerceSessionChatMessages(
  rawMessages: unknown[],
): SessionChatMessage[] {
  return rawMessages
    .map((m) => {
      if (!m || typeof m !== "object") return null;
      const msg = m as Record<string, unknown>;

      const role = typeof msg.role === "string" ? msg.role : null;
      if (!role) return null;

      return {
        role,
        content:
          typeof msg.content === "string"
            ? msg.content
            : msg.content == null
              ? null
              : String(msg.content),
        tool_call_id:
          typeof msg.tool_call_id === "string"
            ? msg.tool_call_id
            : msg.tool_call_id == null
              ? null
              : String(msg.tool_call_id),
        tool_calls: Array.isArray(msg.tool_calls) ? msg.tool_calls : null,
      };
    })
    .filter((m): m is SessionChatMessage => m !== null);
}

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function toToolInput(rawArguments: unknown): unknown {
  if (typeof rawArguments === "string") {
    const trimmed = rawArguments.trim();
    return trimmed ? safeJsonParse(trimmed) : {};
  }
  if (rawArguments && typeof rawArguments === "object") return rawArguments;
  return {};
}

export function convertChatSessionMessagesToUiMessages(
  sessionId: string,
  rawMessages: unknown[],
  options?: { isComplete?: boolean },
): UIMessage<unknown, UIDataTypes, UITools>[] {
  const messages = coerceSessionChatMessages(rawMessages);
  const toolOutputsByCallId = new Map<string, unknown>();

  for (const msg of messages) {
    if (msg.role !== "tool") continue;
    if (!msg.tool_call_id) continue;
    if (msg.content == null) continue;
    toolOutputsByCallId.set(msg.tool_call_id, msg.content);
  }

  const uiMessages: UIMessage<unknown, UIDataTypes, UITools>[] = [];

  messages.forEach((msg, index) => {
    if (msg.role === "tool") return;
    if (msg.role !== "user" && msg.role !== "assistant") return;

    const parts: UIMessage<unknown, UIDataTypes, UITools>["parts"] = [];

    if (typeof msg.content === "string" && msg.content.trim()) {
      parts.push({ type: "text", text: msg.content, state: "done" });
    }

    if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
      for (const rawToolCall of msg.tool_calls) {
        if (!rawToolCall || typeof rawToolCall !== "object") continue;
        const toolCall = rawToolCall as {
          id?: unknown;
          function?: { name?: unknown; arguments?: unknown };
        };

        const toolCallId = String(toolCall.id ?? "").trim();
        const toolName = String(toolCall.function?.name ?? "").trim();
        if (!toolCallId || !toolName) continue;

        const input = toToolInput(toolCall.function?.arguments);
        const output = toolOutputsByCallId.get(toolCallId);

        if (output !== undefined) {
          parts.push({
            type: `tool-${toolName}`,
            toolCallId,
            state: "output-available",
            input,
            output: typeof output === "string" ? safeJsonParse(output) : output,
          });
        } else if (options?.isComplete) {
          // Session is complete (no active stream) but this tool call has
          // no output in the DB — mark as completed to stop stale spinners.
          parts.push({
            type: `tool-${toolName}`,
            toolCallId,
            state: "output-available",
            input,
            output: "",
          });
        } else {
          // Active stream exists: Skip incomplete tool calls during hydration.
          // The resume stream will deliver them fresh with proper SDK state.
          // This prevents "No tool invocation found" errors on page refresh.
          continue;
        }
      }
    }

    if (parts.length === 0) return;

    uiMessages.push({
      id: `${sessionId}-${index}`,
      role: msg.role,
      parts,
    });
  });

  return uiMessages;
}
