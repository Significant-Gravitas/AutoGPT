import type { ChatMessageData } from "../../../ChatMessage/useChatMessage";
import { isAgentOutputResult, isToolOutputPattern } from "../../helpers";

export interface UseMessageItemArgs {
  message: ChatMessageData;
  messages: ChatMessageData[];
  index: number;
  lastAssistantMessageIndex: number;
}

export function useMessageItem({
  message,
  messages,
  index,
  lastAssistantMessageIndex,
}: UseMessageItemArgs) {
  let agentOutput: ChatMessageData | undefined;
  let messageToRender: ChatMessageData = message;

  // Check if assistant message follows a tool_call and looks like a tool output
  if (message.type === "message" && message.role === "assistant") {
    const prevMessage = messages[index - 1];

    // Check if next message is an agent_output tool_response to include in current assistant message
    const nextMessage = messages[index + 1];
    if (
      nextMessage &&
      nextMessage.type === "tool_response" &&
      nextMessage.result
    ) {
      if (isAgentOutputResult(nextMessage.result)) {
        agentOutput = nextMessage;
      }
    }

    // Only convert to tool_response if it follows a tool_call AND looks like a tool output
    if (prevMessage && prevMessage.type === "tool_call") {
      if (isToolOutputPattern(message.content)) {
        // Convert this message to a tool_response format for rendering
        messageToRender = {
          type: "tool_response",
          toolId: prevMessage.toolId,
          toolName: prevMessage.toolName,
          result: message.content,
          success: true,
          timestamp: message.timestamp,
        } as ChatMessageData;
      }
    }
  }

  const isFinalMessage =
    messageToRender.type !== "message" ||
    messageToRender.role !== "assistant" ||
    index === lastAssistantMessageIndex;

  return {
    messageToRender,
    agentOutput,
    isFinalMessage,
  };
}
