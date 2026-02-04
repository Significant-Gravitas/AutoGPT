import type { ChatMessageData } from "../ChatMessage/useChatMessage";

/**
 * Message types hidden after streaming completes.
 * These are operation status messages that shouldn't persist after the final answer.
 *
 * Note: tool_call is NOT included - it shows with an icon and small grey text,
 * which the user wants to keep visible during streaming.
 *
 * Note: tool_response is handled separately - always hidden from main list
 * but shown inside ThinkingAccordion during streaming.
 */
const POST_STREAM_HIDDEN_TYPES = new Set([
  "operation_started",
  "operation_pending",
  "operation_in_progress",
]);

/**
 * Check if a message should be hidden after streaming completes.
 * This includes operation status messages but NOT tool_call (those are always visible).
 */
export function isPostStreamHiddenMessage(msg: ChatMessageData): boolean {
  return POST_STREAM_HIDDEN_TYPES.has(msg.type);
}

/**
 * Check if a message is a tool_response that should be shown in the accordion.
 */
export function isToolResponseMessage(
  msg: ChatMessageData,
): msg is ChatMessageData & { type: "tool_response" } {
  return msg.type === "tool_response";
}

/**
 * Extract tool_response messages from the message list.
 * These are displayed inside the ThinkingAccordion during streaming.
 */
export function extractToolResponses(
  messages: ChatMessageData[],
): Array<ChatMessageData & { type: "tool_response" }> {
  return messages.filter(isToolResponseMessage);
}

/** Type for tool response map */
export type ToolResponseMap = Map<
  string,
  ChatMessageData & { type: "tool_response" }
>;

/**
 * Create a map from toolId to tool_response for efficient lookup.
 * Used to link tool_call messages to their corresponding responses.
 */
export function createToolResponseMap(
  messages: ChatMessageData[],
): ToolResponseMap {
  const map: ToolResponseMap = new Map();
  for (const msg of messages) {
    if (isToolResponseMessage(msg) && msg.toolId) {
      map.set(msg.toolId, msg);
    }
  }
  return map;
}

/**
 * Check if a message is a final assistant response.
 * Final responses are text messages from the assistant role.
 */
export function isFinalAssistantMessage(msg: ChatMessageData): boolean {
  return msg.type === "message" && msg.role === "assistant";
}

/**
 * Check if the message list contains at least one final assistant message.
 * Used to determine if tool chatter can be safely hidden.
 */
export function hasFinalAssistantMessage(messages: ChatMessageData[]): boolean {
  return messages.some(isFinalAssistantMessage);
}

/**
 * Filter messages for display in the main message list.
 *
 * Behavior:
 * - tool_response: ALWAYS hidden from main list (shown in ThinkingAccordion during streaming)
 * - tool_call: ALWAYS visible (shows with icon and small grey text)
 * - operation_*: Hidden after streaming completes with a final answer
 * - Other messages: Always visible
 *
 * This creates a ChatGPT-style UX where:
 * - Tool invocations (tool_call) are always visible (clickable to show response in dialog)
 * - Tool outputs (tool_response) are hidden but viewable via tool_call click
 * - After streaming, user messages, tool calls, and final assistant response remain
 */
export function filterMessagesForDisplay(
  messages: ChatMessageData[],
  isStreaming: boolean,
): ChatMessageData[] {
  // Always filter out tool_response - it's shown in the ThinkingAccordion during streaming
  const filtered = messages.filter((msg) => !isToolResponseMessage(msg));

  // During streaming, keep operation messages visible
  if (isStreaming) {
    return filtered;
  }

  // Check if we have a final assistant message
  const hasFinalAnswer = hasFinalAssistantMessage(messages);

  // If no final answer (error/cancel case), keep operation messages visible for debugging
  if (!hasFinalAnswer) {
    return filtered;
  }

  // Filter out operation messages now that we have a final answer
  // Note: tool_call messages are kept visible (clickable to show tool response)
  return filtered.filter((msg) => !isPostStreamHiddenMessage(msg));
}

export function parseToolResult(
  result: unknown,
): Record<string, unknown> | null {
  try {
    return typeof result === "string"
      ? JSON.parse(result)
      : (result as Record<string, unknown>);
  } catch {
    return null;
  }
}

export function isAgentOutputResult(result: unknown): boolean {
  const parsed = parseToolResult(result);
  return parsed?.type === "agent_output";
}

export function isToolOutputPattern(content: string): boolean {
  const normalizedContent = content.toLowerCase().trim();

  return (
    normalizedContent.startsWith("no agents found") ||
    normalizedContent.startsWith("no results found") ||
    normalizedContent.includes("no agents found matching") ||
    !!normalizedContent.match(/^no \w+ found/i) ||
    (content.length < 150 && normalizedContent.includes("try different")) ||
    (content.length < 200 &&
      !normalizedContent.includes("i'll") &&
      !normalizedContent.includes("let me") &&
      !normalizedContent.includes("i can") &&
      !normalizedContent.includes("i will"))
  );
}

export function formatToolResultValue(result: unknown): string {
  return typeof result === "string"
    ? result
    : result
      ? JSON.stringify(result, null, 2)
      : "";
}

export function findLastMessageIndex(
  messages: ChatMessageData[],
  predicate: (msg: ChatMessageData) => boolean,
): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (predicate(messages[i])) return i;
  }
  return -1;
}

export function shouldSkipAgentOutput(
  message: ChatMessageData,
  prevMessage: ChatMessageData | undefined,
): boolean {
  if (message.type !== "tool_response" || !message.result) return false;

  const isAgentOutput = isAgentOutputResult(message.result);
  return (
    isAgentOutput &&
    !!prevMessage &&
    prevMessage.type === "message" &&
    prevMessage.role === "assistant"
  );
}
