"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState, useMemo } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { CopyIcon, CheckIcon } from "@phosphor-icons/react";

export default function Page() {
  const [input, setInput] = useState("");
  const [copied, setCopied] = useState(false);
  const [sessionId] = useQueryState("sessionId", parseAsString);

  function handleCopySessionId() {
    if (!sessionId) return;
    navigator.clipboard.writeText(sessionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

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
    <SidebarProvider
      defaultOpen={false}
      className="h-[calc(100vh-72px)] min-h-0"
    >
      <ChatSidebar />
      <SidebarInset className="relative flex h-[calc(100vh-80px)] flex-col">
        {sessionId && (
          <div className="absolute flex items-center px-4 py-4">
            <div className="flex items-center gap-2 rounded-3xl border border-neutral-400 bg-neutral-100 px-3 py-1.5 text-sm text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
              <span className="text-xs">{sessionId.slice(0, 8)}...</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={handleCopySessionId}
              >
                {copied ? (
                  <CheckIcon className="h-3.5 w-3.5 text-green-500" />
                ) : (
                  <CopyIcon className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>
          </div>
        )}
        <div className="flex-1 overflow-hidden">
          <ChatContainer
            messages={messages}
            status={status}
            error={error}
            input={input}
            setInput={setInput}
            handleMessageSubmit={handleMessageSubmit}
            onSend={onSend}
          />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
