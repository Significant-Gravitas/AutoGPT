import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { cn } from "@/lib/utils";
import { useCallback } from "react";
import { usePageContext } from "../../usePageContext";
import { ChatInput } from "../ChatInput/ChatInput";
import { MessageList } from "../MessageList/MessageList";
import { useChatContainer } from "./useChatContainer";

export interface ChatContainerProps {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  initialPrompt?: string;
  className?: string;
}

export function ChatContainer({
  sessionId,
  initialMessages,
  initialPrompt,
  className,
}: ChatContainerProps) {
  const { messages, streamingChunks, isStreaming, sendMessage, stopStreaming } =
    useChatContainer({
      sessionId,
      initialMessages,
      initialPrompt,
    });

  const { capturePageContext } = usePageContext();
  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const sendMessageWithContext = useCallback(
    async (content: string, isUserMessage: boolean = true) => {
      const context = capturePageContext();
      await sendMessage(content, isUserMessage, context);
    },
    [sendMessage, capturePageContext],
  );

  return (
    <div
      className={cn(
        "mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-[#f8f8f9]",
        className,
      )}
    >
      {/* Messages - Scrollable */}
      <div className="relative flex min-h-0 flex-1 flex-col">
        <div className="flex min-h-full flex-col justify-end">
          <MessageList
            messages={messages}
            streamingChunks={streamingChunks}
            isStreaming={isStreaming}
            onSendMessage={sendMessageWithContext}
            className="flex-1"
          />
        </div>
      </div>

      {/* Input - Fixed at bottom */}
      <div className="relative px-3 pb-6 pt-2">
        <div className="pointer-events-none absolute top-[-18px] z-10 h-6 w-full bg-gradient-to-b from-transparent to-[#f8f8f9]" />
        <ChatInput
          onSend={sendMessageWithContext}
          disabled={isStreaming || !sessionId}
          isStreaming={isStreaming}
          onStop={stopStreaming}
          placeholder={
            isMobile
              ? "You can search or just ask"
              : 'You can search or just ask — e.g. "create a blog post outline"'
          }
        />
      </div>
    </div>
  );
}
