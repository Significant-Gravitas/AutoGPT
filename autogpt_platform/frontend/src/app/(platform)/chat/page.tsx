"use client";

import { ChatInterface } from "@/components/chat/ChatInterface";

export default function ChatPage() {
  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-4 py-3">
        <h1 className="text-xl font-semibold">AI Agent Discovery Chat</h1>
        <p className="text-sm text-muted-foreground">
          Discover and interact with AI agents through natural conversation
        </p>
      </div>
      <div className="flex-1 overflow-hidden">
        <ChatInterface className="h-full" />
      </div>
    </div>
  );
}
