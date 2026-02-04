"use client";
import { UIDataTypes, UITools, UIMessage } from "ai";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer";
import { EmptySession } from "../EmptySession/EmptySession";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { LayoutGroup, motion } from "framer-motion";

export interface ChatContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  sessionId: string | null;
  isCreatingSession: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string) => void | Promise<void>;
}
export const ChatContainer = ({
  messages,
  status,
  error,
  sessionId,
  isCreatingSession,
  onCreateSession,
  onSend,
}: ChatContainerProps) => {
  const inputLayoutId = "copilot-2-chat-input";

  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <LayoutGroup id="copilot-2-chat-layout">
        <div className="h-full w-full pb-6">
          <div className="flex h-full flex-col">
            {sessionId ? (
              <div className="mx-auto flex h-full w-full max-w-3xl flex-col">
                <ChatMessagesContainer
                  messages={messages}
                  status={status}
                  error={error}
                />
                <motion.div
                  layoutId={inputLayoutId}
                  transition={{ type: "spring", bounce: 0.2, duration: 0.65 }}
                  className="relative px-3 pt-2"
                >
                  <div className="pointer-events-none absolute top-[-18px] z-10 h-6 w-full bg-gradient-to-b from-transparent to-[#f8f8f9] dark:to-background" />
                  <ChatInput
                    inputId="chat-input-session"
                    onSend={onSend}
                    disabled={status === "streaming"}
                    isStreaming={status === "streaming"}
                    onStop={() => {}}
                    placeholder="You can search or just ask"
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
        </div>
      </LayoutGroup>
    </CopilotChatActionsProvider>
  );
};
