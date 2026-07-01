"use client";

import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { useTourCopilot } from "../../useTourCopilot";
import { TourPromptBar } from "../TourPromptBar/TourPromptBar";

interface Props {
  chat: ReturnType<typeof useTourCopilot>;
}

export function TourChatContainer({ chat }: Props) {
  return (
    <CopilotChatActionsProvider onSend={chat.onSend}>
      <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
        <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-[#fafafa]">
          <ChatMessagesContainer
            messages={chat.messages}
            status={chat.status}
            error={chat.error}
            isLoading={false}
            sessionID={chat.sessionId}
            turnStats={chat.turnStats}
            queuedMessages={chat.queuedMessages}
          />
          <div className="relative px-3 pb-3 pt-2">
            <div className="pointer-events-none absolute left-0 right-0 top-[-18px] z-10 h-6 bg-gradient-to-b from-transparent to-[#fafafa]" />
            <TourPromptBar
              key={chat.currentUserPrompt ?? "done"}
              prompt={chat.currentUserPrompt}
              isStreaming={chat.isStreaming}
              isExhausted={chat.isExhausted}
              onSend={() =>
                chat.currentUserPrompt && chat.onSend(chat.currentUserPrompt)
              }
              onReplay={chat.reset}
            />
          </div>
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
