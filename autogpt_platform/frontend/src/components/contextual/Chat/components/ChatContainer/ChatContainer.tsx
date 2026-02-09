import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { GlobeHemisphereEastIcon } from "@phosphor-icons/react";
import { useEffect } from "react";
import { ChatInput } from "../ChatInput/ChatInput";
import { MessageList } from "../MessageList/MessageList";
import { useChatContainer } from "./useChatContainer";

export interface ChatContainerProps {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  initialPrompt?: string;
  className?: string;
  onStreamingChange?: (isStreaming: boolean) => void;
  onOperationStarted?: () => void;
  /** Active stream info from the server for reconnection */
  activeStream?: {
    taskId: string;
    lastMessageId: string;
    operationId: string;
    toolName: string;
  };
}

export function ChatContainer({
  sessionId,
  initialMessages,
  initialPrompt,
  className,
  onStreamingChange,
  onOperationStarted,
  activeStream,
}: ChatContainerProps) {
  const {
    messages,
    streamingChunks,
    isStreaming,
    stopStreaming,
    isRegionBlockedModalOpen,
    sendMessageWithContext,
    handleRegionModalOpenChange,
    handleRegionModalClose,
  } = useChatContainer({
    sessionId,
    initialMessages,
    initialPrompt,
    onOperationStarted,
    activeStream,
  });

  useEffect(() => {
    onStreamingChange?.(isStreaming);
  }, [isStreaming, onStreamingChange]);

  return (
    <div
      className={cn(
        "mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-[#f8f8f9]",
        className,
      )}
    >
      <Dialog
        title={
          <div className="flex items-center gap-2">
            <GlobeHemisphereEastIcon className="size-6" />
            <Text
              variant="body"
              className="text-md font-poppins leading-none md:text-lg"
            >
              Service unavailable
            </Text>
          </div>
        }
        controlled={{
          isOpen: isRegionBlockedModalOpen,
          set: handleRegionModalOpenChange,
        }}
        onClose={handleRegionModalClose}
        styling={{ maxWidth: 550, width: "100%", minWidth: "auto" }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-8">
            <Text variant="body">
              The Autogpt AI model is not available in your region or your
              connection is blocking it. Please try again with a different
              connection.
            </Text>
            <div className="flex justify-center">
              <Button
                type="button"
                variant="primary"
                onClick={handleRegionModalClose}
                className="w-full"
              >
                Got it
              </Button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog>
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
          placeholder="What else can I help with?"
        />
      </div>
    </div>
  );
}
