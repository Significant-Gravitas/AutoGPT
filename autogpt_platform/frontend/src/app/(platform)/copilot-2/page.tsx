"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState, useMemo } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ChatSidebar } from "./tools/components/ChatSidebar/ChatSidebar";

export default function Page() {
  const [sessionId] = useQueryState("sessionId", parseAsString);
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
    transport: transport ?? undefined,
  });

  async function handleStartChat(prompt: string) {
    if (!prompt.trim()) return;
    sendMessage({ text: prompt });
    setInput("");
  }

  // Show landing page when no session exists
  if (!sessionId) {
    return (
      <div className="flex h-full">
        <ChatSidebar isCreating={isCreating} setIsCreating={setIsCreating} />

        <div className="flex h-full flex-1 flex-col items-center justify-center bg-zinc-100 p-4">
          <h2 className="mb-4 text-xl font-semibold text-zinc-700">
            Start a new conversation
          </h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleStartChat(input);
            }}
            className="w-full max-w-md"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isCreating}
              placeholder="Type your message to start..."
              className="w-full rounded-md border border-zinc-300 px-4 py-2"
            />
            <button
              type="submit"
              disabled={isCreating || !input.trim()}
              className="mt-2 w-full rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {isCreating ? "Starting..." : "Start Chat"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <ChatSidebar isCreating={isCreating} setIsCreating={setIsCreating} />

      {/* Chat area */}
      <div className="flex h-full flex-1 flex-col p-4">
        <div className="mb-2 text-sm text-zinc-500">
          Session ID: {sessionId}
        </div>

        <div className="flex-1 overflow-y-auto">
          {messages.map((message) => (
            <div key={message.id} className="flex flex-col gap-4">
              {message.parts.map((part, index) => {
                if (part.type === "tool-find_block") {
                  return (
                    <div
                      key={index}
                      className="w-fit rounded-xl border border-zinc-200 bg-zinc-100 p-2 text-xs text-zinc-700"
                    >
                      {part.state === "input-streaming" && (
                        <p>Finding blocks for you</p>
                      )}
                      {part.state === "input-available" && (
                        <p>
                          Searching blocks for{" "}
                          {(part.input as { query: string }).query}
                        </p>
                      )}
                      {part.state === "output-available" && (
                        <p>
                          Found
                          {
                            (
                              JSON.parse(part.output as string) as {
                                count: number;
                              }
                            ).count
                          }
                          blocks
                        </p>
                      )}
                    </div>
                  );
                } else if (part.type === "text") {
                  return <p key={index}>{part.text}</p>;
                }
              })}
            </div>
          ))}
          {status === "submitted" && <p>Thinking....</p>}
          {error && <div className="text-red-500">Error: {error.message}</div>}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (input.trim()) {
              sendMessage({ text: input });
              setInput("");
            }
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={status !== "ready"}
            placeholder="Say something..."
          />
          <button type="submit" disabled={status !== "ready"}>
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}
