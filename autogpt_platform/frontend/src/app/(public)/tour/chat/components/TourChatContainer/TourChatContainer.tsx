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
    <CopilotChatActionsProvider onSend={chat.onSend} chatSurface="builder">
      <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
        {/* Tour-only card styling. These descendant selectors target the shared
            copilot tool-call markup (.bg-card / .bg-stone-50 / .py-2) by class
            name — if those classes change in ChatMessagesContainer, update them
            here too. Scoped this way to avoid modifying the shared component. */}
        <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-[#fafafa] pb-12 [&_.bg-card]:mb-6 [&_.bg-stone-50]:rounded-xl [&_.bg-stone-50]:border [&_.bg-stone-50]:border-zinc-200/70 [&_.bg-stone-50]:bg-white [&_.bg-stone-50]:!py-2 [&_.bg-stone-50]:shadow-sm [&_.py-2]:py-0">
          <ChatMessagesContainer
            messages={chat.messages}
            status={chat.status}
            error={chat.error}
            isLoading={false}
            sessionID={chat.sessionId}
            turnStats={chat.turnStats}
            queuedMessages={chat.queuedMessages}
          />
          <div className="relative px-3 pb-2 pt-2">
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
