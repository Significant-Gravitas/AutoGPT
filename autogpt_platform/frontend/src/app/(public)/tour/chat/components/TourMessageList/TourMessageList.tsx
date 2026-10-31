"use client";

import { ThinkingIndicator } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/ThinkingIndicator";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/organisms/Conversation/Conversation";
import {
  Message,
  MessageContent,
} from "@/components/organisms/Message/Message";
import type { TourMessage, TourPart } from "../../script/types";
import { TourAgentCard } from "../TourAgentCard/TourAgentCard";
import { TourArtifactCard } from "../TourArtifactCard/TourArtifactCard";
import { TourPlanCard } from "../TourPlanCard/TourPlanCard";
import { TourStreamingText } from "./TourStreamingText";

// Mirrors the real copilot's MessageContent overrides (purple user bubble,
// 1rem text) so the tour chat reads exactly like the product.
const MESSAGE_CONTENT_CLASSES =
  // oxlint-disable-next-line design-system/no-restyle -- the copilot's purple user bubble; MessageContent has no variant for it
  "text-body-lg leading-relaxed group-[.is-user]:rounded-xl group-[.is-user]:rounded-br-none group-[.is-user]:bg-accent-surface group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-foreground-strong group-[.is-assistant]:text-foreground-strong";

const CARD_ANIMATION_CLASSES =
  "animate-in fade-in slide-in-from-bottom-2 fill-mode-both duration-500";

interface Props {
  messages: TourMessage[];
  isStreaming: boolean;
  /** Rendered inside the conversation scroll area after the transcript —
   * the demo end card lives here so it reads as part of the chat flow. */
  footer?: React.ReactNode;
}

export function TourMessageList({ messages, isStreaming, footer }: Props) {
  const last = messages[messages.length - 1];
  const showThinking =
    isStreaming && last?.role === "assistant" && last.parts.length === 0;

  return (
    <Conversation className="min-h-0 flex-1">
      {/* oxlint-disable-next-line design-system/no-restyle -- the tour thread's tighter spacing; ConversationContent has no size for it */}
      <ConversationContent className="gap-6 px-3 pt-4 pb-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className="duration-300 animate-in fade-in slide-in-from-bottom-2"
          >
            <Message from={message.role}>
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
          </div>
        ))}
        {/* The demo has no backend status stream, so it names the state
            itself — the indicator shows no label otherwise. */}
        {showThinking && (
          <ThinkingIndicator
            active
            elapsedSeconds={0}
            statusMessage="Thinking…"
          />
        )}
        {footer}
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
