import { cn } from "@/lib/utils";
import { ChatMessage } from "../ChatMessage/ChatMessage";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { StreamingMessage } from "../StreamingMessage/StreamingMessage";
import { useMessageList } from "./useMessageList";

export interface MessageListProps {
  messages: ChatMessageData[];
  streamingChunks?: string[];
  isStreaming?: boolean;
  className?: string;
  onStreamComplete?: () => void;
  onSendMessage?: (content: string) => void;
}

export function MessageList({
  messages,
  streamingChunks = [],
  isStreaming = false,
  className,
  onStreamComplete,
  onSendMessage,
}: MessageListProps) {
  const { messagesEndRef, messagesContainerRef } = useMessageList({
    messageCount: messages.length,
    isStreaming,
  });

  return (
    <div
      ref={messagesContainerRef}
      className={cn(
        "flex-1 overflow-y-auto",
        "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 dark:scrollbar-thumb-zinc-700",
        className,
      )}
    >
      <div className="space-y-0">
        {/* Render all persisted messages */}
        {messages.map((message, index) => (
          <ChatMessage
            key={index}
            message={message}
            onSendMessage={onSendMessage}
          />
        ))}

        {/* Render streaming message if active */}
        {isStreaming && streamingChunks.length > 0 && (
          <StreamingMessage
            chunks={streamingChunks}
            onComplete={onStreamComplete}
          />
        )}

        {/* Invisible div to scroll to */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
