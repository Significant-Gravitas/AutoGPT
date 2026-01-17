import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef } from "react";
import { usePageContext } from "../../usePageContext";
import { ChatInput } from "../ChatInput/ChatInput";
import { MessageList } from "../MessageList/MessageList";
import { QuickActionsWelcome } from "../QuickActionsWelcome/QuickActionsWelcome";
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

  const quickActions = [
    "Find agents for social media management",
    "Show me agents for content creation",
    "Help me automate my business",
    "What can you help me with?",
  ];

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col", className)}
      style={{
        backgroundColor: "#ffffff",
        backgroundImage:
          "radial-gradient(#e5e5e5 0.5px, transparent 0.5px), radial-gradient(#e5e5e5 0.5px, #ffffff 0.5px)",
        backgroundSize: "20px 20px",
        backgroundPosition: "0 0, 10px 10px",
      }}
    >
      {/* Messages or Welcome Screen */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden pb-24">
        {messages.length === 0 ? (
          <QuickActionsWelcome
            title="Welcome to AutoGPT Copilot"
            description="Start a conversation to discover and run AI agents."
            actions={quickActions}
            onActionClick={sendMessageWithContext}
            disabled={isStreaming || !sessionId}
          />
        ) : (
          <MessageList
            messages={messages}
            streamingChunks={streamingChunks}
            isStreaming={isStreaming}
            onSendMessage={sendMessageWithContext}
            className="flex-1"
          />
        )}
      </div>

      {/* Input - Always visible */}
      <div className="sticky bottom-0 z-50 border-t border-zinc-200 bg-white p-4">
        <ChatInput
          onSend={sendMessageWithContext}
          disabled={isStreaming || !sessionId}
          placeholder={
            sessionId ? "Type your message..." : "Creating session..."
          }
        />
      </div>
    </div>
  );
}
