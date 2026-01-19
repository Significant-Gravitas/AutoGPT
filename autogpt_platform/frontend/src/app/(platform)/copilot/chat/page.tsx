"use client";

import { Chat } from "@/components/contextual/Chat/Chat";
import { useCopilotChatPage } from "./useCopilotChatPage";

export default function CopilotChatPage() {
  const { isFlagReady, isChatEnabled, sessionId, prompt } =
    useCopilotChatPage();

  if (!isFlagReady || isChatEnabled === false) {
    return null;
  }

  return (
    <div className="flex h-full flex-col">
      <Chat
        className="flex-1"
        urlSessionId={sessionId}
        initialPrompt={prompt}
        showNewChatButton={false}
      />
    </div>
  );
}
