import {
    Conversation,
    ConversationContent,
    ConversationEmptyState,
    ConversationScrollButton,
  } from "@/components/ai-elements/conversation";
  import {
    Message,
    MessageContent,
    MessageResponse,
  } from "@/components/ai-elements/message";
import { MessageSquareIcon } from "lucide-react";
import { UIMessage, UIDataTypes, UITools } from "ai";


interface ChatMessagesContainerProps {
    messages: UIMessage<unknown, UIDataTypes, UITools>[];
    status: string;
    error: Error | undefined;
    handleSubmit: (e: React.FormEvent) => void;
    input: string;
    setInput: (input: string) => void;
  }

export const ChatMessagesContainer = ({messages, status, error, handleSubmit, input, setInput}: ChatMessagesContainerProps) => {
  return (
    <div className="flex h-full flex-1 flex-col">
        <Conversation className="flex-1">
          <ConversationContent>
            {messages.length === 0 ? (
              <ConversationEmptyState
                icon={<MessageSquareIcon className="size-12" />}
                title="Start a conversation"
                description="Type a message below to begin chatting"
              />
            ) : (
              messages.map((message) => (
                <Message from={message.role} key={message.id}>
                  <MessageContent>
                    {message.parts.map((part, i) => {
                      switch (part.type) {
                        case "text":
                          return (
                            <MessageResponse key={`${message.id}-${i}`}>
                              {part.text}
                            </MessageResponse>
                          );
                        case "tool-find_block":
                          return (
                            <div
                              key={`${message.id}-${i}`}
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
                                  Found{" "}
                                  {
                                    (
                                      JSON.parse(part.output as string) as {
                                        count: number;
                                      }
                                    ).count
                                  }{" "}
                                  blocks
                                </p>
                              )}
                            </div>
                          );
                        default:
                          return null;
                      }
                    })}
                  </MessageContent>
                </Message>
              ))
            )}
            {status === "submitted" && (
              <Message from="assistant">
                <MessageContent>
                  <p className="text-zinc-500">Thinking...</p>
                </MessageContent>
              </Message>
            )}
            {error && (
              <div className="rounded-lg bg-red-50 p-3 text-red-600">
                Error: {error.message}
              </div>
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <form onSubmit={handleSubmit} className="border-t p-4">
          <div className="mx-auto flex max-w-2xl gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={status !== "ready"}
              placeholder="Say something..."
              className="flex-1 rounded-md border border-zinc-300 px-4 py-2 focus:border-zinc-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={status !== "ready" || !input.trim()}
              className="rounded-md bg-zinc-900 px-4 py-2 text-white transition-colors hover:bg-zinc-800 disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </form>
      </div>
  );
};  