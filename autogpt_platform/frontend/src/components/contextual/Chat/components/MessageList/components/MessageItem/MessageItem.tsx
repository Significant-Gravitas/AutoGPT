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
}

export function MessageItem({
  message,
  messages,
  index,
  lastAssistantMessageIndex,
  isStreaming = false,
  onSendMessage,
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
    />
  );
}
