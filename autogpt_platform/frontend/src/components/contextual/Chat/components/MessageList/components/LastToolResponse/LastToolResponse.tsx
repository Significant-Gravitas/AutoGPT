import { AIChatBubble } from "../../../AIChatBubble/AIChatBubble";
import type { ChatMessageData } from "../../../ChatMessage/useChatMessage";
import { MarkdownContent } from "../../../MarkdownContent/MarkdownContent";
import { formatToolResponse } from "../../../ToolResponseMessage/helpers";
import { shouldSkipAgentOutput } from "../../helpers";

export interface LastToolResponseProps {
  message: ChatMessageData;
  prevMessage: ChatMessageData | undefined;
}

export function LastToolResponse({
  message,
  prevMessage,
}: LastToolResponseProps) {
  if (message.type !== "tool_response") return null;

  // Skip if this is an agent_output that should be rendered inside assistant message
  if (shouldSkipAgentOutput(message, prevMessage)) return null;

  const formattedText = formatToolResponse(message.result, message.toolName);

  return (
    <div className="min-w-0 overflow-x-hidden hyphens-auto break-words px-4 py-2">
      <AIChatBubble>
        <MarkdownContent content={formattedText} />
      </AIChatBubble>
    </div>
  );
}
