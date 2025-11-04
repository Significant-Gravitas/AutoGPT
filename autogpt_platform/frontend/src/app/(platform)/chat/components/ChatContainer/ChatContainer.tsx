import { cn } from "@/lib/utils";
import { ChatInput } from "@/components/atoms/ChatInput/ChatInput";
import { MessageList } from "@/components/molecules/MessageList/MessageList";
import { QuickActionsWelcome } from "@/components/molecules/QuickActionsWelcome/QuickActionsWelcome";
import { useChatContainer } from "./useChatContainer";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";

export interface ChatContainerProps {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  onRefreshSession: () => Promise<void>;
  className?: string;
}

export function ChatContainer({
  sessionId,
  initialMessages,
  onRefreshSession,
  className,
}: ChatContainerProps) {
  const { messages, streamingChunks, isStreaming, sendMessage } =
    useChatContainer({
      sessionId,
      initialMessages,
      onRefreshSession,
    });

  const quickActions = [
    "Find agents for data analysis",
    "Show me automation agents",
    "Help me build a workflow",
    "What can you help me with?",
  ];

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Messages or Welcome Screen */}
      {messages.length === 0 ? (
        <QuickActionsWelcome
          title="Welcome to AutoGPT Chat"
          description="Start a conversation to discover and run AI agents."
          actions={quickActions}
          onActionClick={sendMessage}
          disabled={isStreaming || !sessionId}
        />
      ) : (
        <MessageList
          messages={messages}
          streamingChunks={streamingChunks}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
          className="flex-1"
        />
      )}

      {/* Input - Always visible */}
      <div className="border-t border-zinc-200 p-4 dark:border-zinc-800">
        <ChatInput
          onSend={sendMessage}
          disabled={isStreaming || !sessionId}
          placeholder={
            sessionId ? "Type your message..." : "Creating session..."
          }
        />
      </div>
    </div>
  );
}
