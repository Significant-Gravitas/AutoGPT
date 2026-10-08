import {
  ConversationBubble,
  ConversationContent,
} from "@/components/ui/conversation";
import {
  Message,
  MessageAvatar,
  MessageContent,
} from "@/components/ui/message";
import type { ReactNode } from "react";

interface Props {
  from: "user" | "agent";
  avatar?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function ChatMessage({ from, avatar, children, className }: Props) {
  if (from === "user") {
    return (
      <Message align="end" data-from="user" className={className}>
        <MessageContent>
          <ConversationBubble variant="secondary" align="end">
            <ConversationContent className="text-base leading-relaxed whitespace-pre-wrap">
              {children}
            </ConversationContent>
          </ConversationBubble>
        </MessageContent>
      </Message>
    );
  }

  return (
    <Message align="start" data-from="agent" className={className}>
      {avatar ? (
        <MessageAvatar className="min-w-0 self-start bg-transparent">
          {avatar}
        </MessageAvatar>
      ) : null}
      <MessageContent className="text-base leading-relaxed text-foreground">
        {children}
      </MessageContent>
    </Message>
  );
}
