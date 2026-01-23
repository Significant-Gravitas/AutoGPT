import type { ChatMessageData } from "../ChatMessage/useChatMessage";

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
