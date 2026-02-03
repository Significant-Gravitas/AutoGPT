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
import { UIMessage, UIDataTypes, UITools, ToolUIPart } from "ai";
import { FindBlocksTool } from "../../tools/FindBlocks/FindBlocks";

interface ChatMessagesContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  handleSubmit: (e: React.FormEvent) => void;
  input: string;
  setInput: (input: string) => void;
}

export const ChatMessagesContainer = ({
  messages,
  status,
  error,
}: ChatMessagesContainerProps) => {
  return (
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
              <MessageContent
                className={
                  "rounded-xl border px-3 py-2 " +
                  "group-[.is-user]:rounded-2xl group-[.is-user]:border-purple-200 group-[.is-user]:bg-purple-100 group-[.is-user]:text-slate-900 " +
                  "group-[.is-assistant]:border-none group-[.is-assistant]:bg-slate-50/20 group-[.is-assistant]:text-slate-900"
                }
              >
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
                        <FindBlocksTool
                          key={`${message.id}-${i}`}
                          part={part as ToolUIPart}
                        />
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
  );
};
