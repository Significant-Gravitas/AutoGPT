import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef } from "react";
import { usePageContext } from "../../usePageContext";
import { ChatInput } from "../ChatInput/ChatInput";
import { MessageList } from "../MessageList/MessageList";
import { useChatContainer } from "./useChatContainer";

export interface ChatContainerProps {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  className?: string;
  initialPrompt?: string | null;
}

export function ChatContainer({
  sessionId,
  initialMessages,
  className,
  initialPrompt,
}: ChatContainerProps) {
  const { messages, streamingChunks, isStreaming, sendMessage } =
    useChatContainer({
      sessionId,
      initialMessages,
    });
  const { capturePageContext } = usePageContext();
  const hasSentInitialRef = useRef(false);

  // Wrap sendMessage to automatically capture page context
  const sendMessageWithContext = useCallback(
    async (content: string, isUserMessage: boolean = true) => {
      const context = capturePageContext();
      await sendMessage(content, isUserMessage, context);
    },
    [sendMessage, capturePageContext],
  );

  useEffect(
    function handleInitialPrompt() {
      if (!initialPrompt) return;
      if (hasSentInitialRef.current) return;
      if (!sessionId) return;
      if (messages.length > 0) return;
      hasSentInitialRef.current = true;
      void sendMessageWithContext(initialPrompt);
    },
    [initialPrompt, messages.length, sendMessageWithContext, sessionId],
  );

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col max-w-3xl mx-auto bg-[#f8f8f9]", className)}
    >
      {/* Messages or Welcome Screen - Scrollable */}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto relative">
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
      <div className="relative pb-4 pt-2">
        <div className="absolute w-full top-[-18px] h-6 pointer-events-none bg-gradient-to-b from-transparent to-[#f8f8f9] z-10" />
        <ChatInput
          onSend={sendMessageWithContext}
          disabled={isStreaming || !sessionId}
          placeholder="You can search or just ask — e.g. “create a blog post outline”"
        />
      </div>
    </div>
  );
}
