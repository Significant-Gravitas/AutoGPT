"use client";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { LayoutGroup, motion } from "framer-motion";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { EmptySession } from "../EmptySession/EmptySession";

export interface ChatContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  sessionId: string | null;
  isLoadingSession: boolean;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string) => void | Promise<void>;
  onStop: () => void;
}
export const ChatContainer = ({
  messages,
  status,
  error,
  sessionId,
  isLoadingSession,
  isCreatingSession,
  onCreateSession,
  onSend,
  onStop,
}: ChatContainerProps) => {
  const inputLayoutId = "copilot-2-chat-input";

  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <LayoutGroup id="copilot-2-chat-layout">
        <div className="flex h-full min-h-0 w-full flex-col bg-[#f8f8f9] px-2 lg:px-0">
          {sessionId ? (
            <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col">
              <ChatMessagesContainer
                messages={messages}
                status={status}
                error={error}
                isLoading={isLoadingSession}
              />
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="relative px-3 pb-2 pt-2"
              >
                <div className="pointer-events-none absolute left-0 right-0 top-[-18px] z-10 h-6 bg-gradient-to-b from-transparent to-[#f8f8f9]" />
                <ChatInput
                  inputId="chat-input-session"
                  onSend={onSend}
                  disabled={status === "streaming"}
                  isStreaming={status === "streaming"}
                  onStop={onStop}
                  placeholder="What else can I help with?"
                />
              </motion.div>
            </div>
          ) : (
            <EmptySession
              inputLayoutId={inputLayoutId}
              isCreatingSession={isCreatingSession}
              onCreateSession={onCreateSession}
              onSend={onSend}
            />
          )}
        </div>
      </LayoutGroup>
    </CopilotChatActionsProvider>
  );
};
