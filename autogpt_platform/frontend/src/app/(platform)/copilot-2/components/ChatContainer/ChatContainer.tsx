"use client"
import { UIDataTypes, UITools, UIMessage } from "ai";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer"
import { EmptySession } from "../EmptySession/EmptySession";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { useState } from "react";
import { parseAsString, useQueryState } from "nuqs";

export interface ChatContainerProps {
    messages: UIMessage<unknown, UIDataTypes, UITools>[];
    status: string;
    error: Error | undefined;
    input: string;
    setInput: (input: string) => void;
    handleMessageSubmit: (e: React.FormEvent) => void;
    onSend: (message: string) => void;
}
export const ChatContainer = ({messages, status, error, input, setInput, handleMessageSubmit, onSend}: ChatContainerProps) => {
    const [isCreating, setIsCreating] = useState(false);
    const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);

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
    

    return (
        <div className="mx-auto h-[calc(100vh-60px)] w-full max-w-3xl pb-6">
        <div className="flex h-full flex-col">
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
            <EmptySession
              isCreating={isCreating}
              onCreateSession={createSession}
            />
          )}
                <div className="relative px-3  pt-2">
        <div className="pointer-events-none absolute top-[-18px] z-10 h-6 w-full bg-gradient-to-b from-transparent to-[#f8f8f9]" />
        <ChatInput
          onSend={onSend}
          disabled={status === "streaming" || !sessionId}
          isStreaming={status === "streaming"}
          onStop={() => {}}
          placeholder="You can search or just ask"
        />
      </div>
        </div>
      </div>
    )
}