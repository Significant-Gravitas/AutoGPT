"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState, useMemo } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ChatSidebar } from "./components/ChatSidebar/ChatSidebar";
import { EmptySession } from "./components/EmptySession/EmptySession";
import { ChatMessagesContainer } from "./components/ChatMessagesContainer/ChatMessagesContainer";
import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";

export default function Page() {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const [isCreating, setIsCreating] = useState(false);
  const [input, setInput] = useState("");

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

  async function createSession(e: React.FormEvent) {
    e.preventDefault();
    if (isCreating) return;
    setIsCreating(true);
    try {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status === 200 && response.data?.id) {
        setSessionId(response.data.id);
      }
    } finally {
      setIsCreating(false);
    }
  }

  function handleMessageSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !sessionId) return;

    sendMessage({ text: input });
    setInput("");
  }

  return (
    <div className="flex h-full">
      <ChatSidebar isCreating={isCreating} setIsCreating={setIsCreating} />

      {sessionId ? (
        <ChatMessagesContainer
          messages={messages}
          status={status}
          error={error}
          handleSubmit={handleMessageSubmit}
          input={input}
          setInput={setInput}
        />
      ) : (
        <EmptySession isCreating={isCreating} onCreateSession={createSession} />
      )}
    </div>
  );
}
