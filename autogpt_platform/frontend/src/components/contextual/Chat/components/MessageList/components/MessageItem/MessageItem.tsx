import type { ToolResponseMap } from "../../helpers";
import { ChatMessage } from "../../../ChatMessage/ChatMessage";
import type { ChatMessageData } from "../../../ChatMessage/useChatMessage";
import { useMessageItem } from "./useMessageItem";

export interface MessageItemProps {
  message: ChatMessageData;
  messages: ChatMessageData[];
  index: number;
  lastAssistantMessageIndex: number;
  isStreaming?: boolean;
  onSendMessage?: (content: string) => void;
  /** Map from toolId to tool_response for linking tool calls to their responses */
  toolResponseMap?: ToolResponseMap;
}

export function MessageItem({
  message,
  messages,
  index,
  lastAssistantMessageIndex,
  isStreaming = false,
  onSendMessage,
  toolResponseMap,
}: MessageItemProps) {
  const { messageToRender, agentOutput, isFinalMessage } = useMessageItem({
    message,
    messages,
    index,
    lastAssistantMessageIndex,
  });

  return (
    <ChatMessage
      message={messageToRender}
      messages={messages}
      index={index}
      isStreaming={isStreaming}
      onSendMessage={onSendMessage}
      agentOutput={agentOutput}
      isFinalMessage={isFinalMessage}
      toolResponseMap={toolResponseMap}
    />
  );
}
