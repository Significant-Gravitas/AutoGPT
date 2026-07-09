"use client";

import { ThinkingIndicator } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/ThinkingIndicator";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import type { TourMessage, TourPart } from "../../script/types";
import { TourAgentCard } from "../TourAgentCard/TourAgentCard";
import { TourArtifactCard } from "../TourArtifactCard/TourArtifactCard";
import { TourPlanCard } from "../TourPlanCard/TourPlanCard";
import { TourStreamingText } from "./TourStreamingText";

// Mirrors the real copilot's MessageContent overrides (purple user bubble,
// 1rem text) so the tour chat reads exactly like the product.
const MESSAGE_CONTENT_CLASSES =
  "text-base leading-relaxed " +
  "group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0] " +
  "group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900";

const CARD_ANIMATION_CLASSES =
  "animate-in fade-in slide-in-from-bottom-2 fill-mode-both duration-500";

interface Props {
  messages: TourMessage[];
  isStreaming: boolean;
}

export function TourMessageList({ messages, isStreaming }: Props) {
  const last = messages[messages.length - 1];
  const showThinking =
    isStreaming && last?.role === "assistant" && last.parts.length === 0;

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="gap-6 px-3 pb-6 pt-4">
        {messages.map((message) => (
          <Message
            key={message.id}
            from={message.role}
            className="duration-300 animate-in fade-in slide-in-from-bottom-2"
          >
            <MessageContent className={MESSAGE_CONTENT_CLASSES}>
              {message.parts.map((part, index) => (
                <TourPartRenderer
                  key={`${message.id}-${index}`}
                  part={part}
                  role={message.role}
                  hasFollowingPart={index < message.parts.length - 1}
                  followsCard={
                    index > 0 && message.parts[index - 1].type !== "text"
                  }
                />
              ))}
            </MessageContent>
          </Message>
        ))}
        {showThinking && <ThinkingIndicator active elapsedSeconds={0} />}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}

function TourPartRenderer({
  part,
  role,
  hasFollowingPart,
  followsCard,
}: {
  part: TourPart;
  role: TourMessage["role"];
  hasFollowingPart: boolean;
  followsCard: boolean;
}) {
  const textSpacing = followsCard ? "mt-3" : undefined;

  switch (part.type) {
    case "text":
      if (role === "user") return <p className={textSpacing}>{part.text}</p>;
      return (
        <div className={textSpacing}>
          <TourStreamingText text={part.text} />
        </div>
      );
    case "plan":
      return (
        <div className={CARD_ANIMATION_CLASSES}>
          <TourPlanCard plan={part.plan} />
        </div>
      );
    case "agent":
      return (
        <div className={CARD_ANIMATION_CLASSES}>
          <TourAgentCard agent={part.agent} runCompleted={hasFollowingPart} />
        </div>
      );
    case "artifact":
      return (
        <div className={CARD_ANIMATION_CLASSES}>
          <TourArtifactCard artifact={part.artifact} />
        </div>
      );
    default: {
      const exhaustiveCheck: never = part;
      return exhaustiveCheck;
    }
  }
}
