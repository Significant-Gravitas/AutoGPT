"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState, useMemo } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { EmptySession } from "./components/EmptySession/EmptySession";
import { ChatMessagesContainer } from "./components/ChatMessagesContainer/ChatMessagesContainer";
import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { useSearchParams } from "next/navigation";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";

export default function Page() {
  const [input, setInput] = useState("");
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId") ?? undefined;

  const transport = useMemo(() => {
    if (!sessionId) return null;
    return new DefaultChatTransport({
      api: `/api/chat/sessions/${sessionId}/stream`,
      prepareSendMessagesRequest: ({ messages }) => {
        const last = messages[messages.length - 1];
        return {
          body: {
            message: last.parts
              ?.map((p) => (p.type === "text" ? p.text : ""))
              .join(""),
            is_user_message: last.role === "user",
            context: null,
          },
        };
      },
    });
  }, [sessionId]);

  const { messages, sendMessage, status, error } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
  });


  function handleMessageSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !sessionId) return;

    sendMessage({ text: input });
    setInput("");
  }

  function onSend(message: string) {
    sendMessage({ text: message });
  }

  return (
    <SidebarProvider className="min-h-0 h-[calc(100vh-60px)]">
      <ChatSidebar />
      <SidebarInset className="h-[calc(100vh-60px)]">
        <ChatContainer
          messages={messages}
          status={status}
          error={error}
          input={input}
          setInput={setInput}
          handleMessageSubmit={handleMessageSubmit}
          onSend={onSend}
        />
      </SidebarInset>
    </SidebarProvider>
  );
}
