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
 * Deduplicate consecutive tool_call messages with the same toolName.
 * Shows only the last one from each consecutive group.
 */
function deduplicateConsecutiveToolCalls(
  messages: ChatMessageData[],
): ChatMessageData[] {
  const result: ChatMessageData[] = [];

  for (let i = 0; i < messages.length; i++) {
    const current = messages[i];

    // If not a tool_call, just add it
    if (current.type !== "tool_call") {
      result.push(current);
      continue;
    }

    // Check if there's a next message that's also a tool_call with the same toolName
    const next = messages[i + 1];
    if (
      next &&
      next.type === "tool_call" &&
      "toolName" in current &&
      "toolName" in next &&
      current.toolName === next.toolName
    ) {
      // Skip this one, the next one (or a later one) will be shown
      continue;
    }

    // This is the last in a consecutive group (or standalone), add it
    result.push(current);
  }

  return result;
}

/**
 * Filter messages for display in the main message list.
 *
 * Behavior:
 * - tool_response: ALWAYS hidden from main list (errors shown inline in tool_call)
 * - tool_call: ALWAYS visible (shows with icon and small grey text)
 *   EXCEPT: Consecutive tool_calls with same toolName are deduplicated (show only last)
 * - operation_*: Hidden after streaming completes with a final answer
 * - Other messages: Always visible
 *
 * This creates a ChatGPT-style UX where:
 * - Tool invocations (tool_call) are always visible (clickable to show response in dialog)
 * - Tool outputs (tool_response) are hidden but viewable via tool_call click
 * - Tool errors are shown inline below the tool_call message
 * - After streaming, user messages, tool calls, and final assistant response remain
 */
export function filterMessagesForDisplay(
  messages: ChatMessageData[],
  isStreaming: boolean,
): ChatMessageData[] {
  // Always filter out tool_response - errors are shown inline in tool_call
  const filtered = messages.filter((msg) => !isToolResponseMessage(msg));

  // Deduplicate consecutive tool_call messages with the same toolName
  const deduplicated = deduplicateConsecutiveToolCalls(filtered);

  // During streaming, keep operation messages visible
  if (isStreaming) {
    return deduplicated;
  }

  // Check if we have a final assistant message
  const hasFinalAnswer = hasFinalAssistantMessage(messages);

  const assistantMsgCount = messages.filter(
    (m) => m.type === "message" && m.role === "assistant",
  ).length;

  console.log("[filterMessagesForDisplay] Post-streaming filter", {
    isStreaming,
    hasFinalAnswer,
    totalMessages: messages.length,
    assistantMessages: assistantMsgCount,
  });

  // If no final answer (error/cancel case), keep operation messages visible for debugging
  if (!hasFinalAnswer) {
    console.log(
      "[filterMessagesForDisplay] No final answer - keeping operation messages",
    );
    return deduplicated;
  }

  // Filter out operation messages now that we have a final answer
  // Note: tool_call messages are kept visible (clickable to show tool response)
  return deduplicated.filter((msg) => !isPostStreamHiddenMessage(msg));
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
