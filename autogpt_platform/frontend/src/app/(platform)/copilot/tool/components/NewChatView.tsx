"use client";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { ThinkingIndicator } from "../../components/ChatMessagesContainer/components/ThinkingIndicator";
import type { MessagePart } from "../../components/ChatMessagesContainer/helpers";
import { useElapsedTimer } from "../../components/JobStatsBar/useElapsedTimer";
import { ToolChain } from "../../components/ToolChain/ToolChain";
import { buildChainSegments } from "../../components/ToolChain/helpers";
import type { ChatMessage } from "../helpers";
import { StreamingText } from "./StreamingText";

interface Props {
  messages: ChatMessage[];
  status: "ready" | "streaming";
  statusMessage: string | null;
}

function AssistantParts({
  parts,
  isStreamingMessage,
}: {
  parts: MessagePart[];
  isStreamingMessage: boolean;
}) {
  const segments = buildChainSegments(parts);
  return (
    <>
      {segments.map((segment, i) => {
        const isLastSegment = i === segments.length - 1;
        if (segment.kind === "chain") {
          return (
            <ToolChain
              key={`chain-${segment.index}`}
              parts={segment.parts}
              isStreaming={isStreamingMessage && isLastSegment}
            />
          );
        }
        if (segment.part.type === "text") {
          // Live tail types with a caret; swaps to markdown once superseded.
          if (isStreamingMessage && isLastSegment) {
            return (
              <StreamingText
                key={`part-${segment.index}`}
                text={segment.part.text}
              />
            );
          }
          return (
            <MessageResponse key={`part-${segment.index}`}>
              {segment.part.text}
            </MessageResponse>
          );
        }
        return null;
      })}
    </>
  );
}

export function NewChatView({ messages, status, statusMessage }: Props) {
  const isStreaming = status === "streaming";
  const { elapsedSeconds } = useElapsedTimer(isStreaming);

  return (
    <Conversation resize="instant" className="min-h-0 flex-1">
      <ConversationContent className="flex min-h-full flex-1 flex-col gap-6 px-6 py-4">
        {messages.map((message, messageIndex) => {
          const isStreamingMessage =
            status === "streaming" &&
            messageIndex === messages.length - 1 &&
            message.role === "assistant";

          if (message.role === "user") {
            return (
              <Message from="user" key={message.id}>
                <MessageContent className="rounded-xl bg-purple-100 px-3 py-2.5 text-[1rem] leading-relaxed text-slate-900 [border-bottom-right-radius:0]">
                  {(message.parts as MessagePart[]).map((part, i) =>
                    part.type === "text" ? (
                      <span key={i}>{part.text}</span>
                    ) : null,
                  )}
                </MessageContent>
              </Message>
            );
          }

          return (
            <Message from="assistant" key={message.id}>
              <MessageContent className="bg-transparent text-[1rem] leading-relaxed text-slate-900">
                <AssistantParts
                  parts={message.parts as MessagePart[]}
                  isStreamingMessage={isStreamingMessage}
                />
              </MessageContent>
            </Message>
          );
        })}
        {isStreaming && (
          <ThinkingIndicator
            active
            elapsedSeconds={elapsedSeconds}
            statusMessage={statusMessage}
            showTimeAfterSeconds={0}
          />
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
