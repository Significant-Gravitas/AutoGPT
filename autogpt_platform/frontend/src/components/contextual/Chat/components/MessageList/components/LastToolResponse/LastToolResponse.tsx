import type { ChatMessageData } from "../../../ChatMessage/useChatMessage";
import { ToolResponseMessage } from "../../../ToolResponseMessage/ToolResponseMessage";
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

  if (shouldSkipAgentOutput(message, prevMessage)) return null;

  return (
    <div className="min-w-0 overflow-x-hidden hyphens-auto break-words px-4 py-2">
      <ToolResponseMessage
        toolId={message.toolId}
        toolName={message.toolName}
        result={message.result}
      />
    </div>
  );
}
