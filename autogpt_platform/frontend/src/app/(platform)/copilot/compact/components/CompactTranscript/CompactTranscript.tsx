"use client";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Text } from "@/components/atoms/Text/Text";
import { AgentStatusAvatar } from "@/components/molecules/AgentStatusAvatar/AgentStatusAvatar";
import {
  type AgentStatus,
  isAgentBusy,
} from "@/components/molecules/AgentStatusAvatar/helpers";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import { ChatMessage } from "@/components/molecules/ChatMessage/ChatMessage";
import type { UIMessage } from "ai";
import { countHeldCalls } from "../../../components/ChatMessagesContainer/heldCallRows";
import { extractReviewTarget } from "../../../components/ChatMessagesContainer/helpers";
import { CopilotPendingReviews } from "../../../components/CopilotPendingReviews/CopilotPendingReviews";
import { buildCompactBlocks, getUserText } from "../../helpers";
import { AgentTurn } from "../AgentTurn/AgentTurn";

interface Props {
  sessionId: string | null;
  messages: UIMessage[];
  agentStatus: AgentStatus;
  agentActivity: string;
  isStreaming: boolean;
  queuedMessages: string[];
}

export function CompactTranscript({
  sessionId,
  messages,
  agentStatus,
  agentActivity,
  isStreaming,
  queuedMessages,
}: Props) {
  const last = messages.at(-1);
  const isTailEmpty =
    !last ||
    last.role !== "assistant" ||
    buildCompactBlocks(last.parts).length === 0;
  const showPendingRow = isAgentBusy(agentStatus) && isTailEmpty;
  const reviewTarget = extractReviewTarget(messages);

  if (messages.length === 0 && !showPendingRow) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
        <AgentStatusAvatar
          status={agentStatus}
          name={AUTOPILOT_NAME}
          src={AUTOPILOT_AVATAR_URL}
          size="lg"
          className="size-16"
        />
        <Text variant="h4" as="h1">
          What can I take off your plate?
        </Text>
      </div>
    );
  }

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="mx-auto w-full max-w-2xl gap-6 px-4 py-6">
        {messages.map((message, index) =>
          message.role === "user" ? (
            <ChatMessage key={message.id} from="user">
              {getUserText(message)}
            </ChatMessage>
          ) : (
            <ChatMessage key={message.id} from="agent">
              <AgentTurn
                message={message}
                isStreaming={isStreaming && index === messages.length - 1}
              />
            </ChatMessage>
          ),
        )}
        {showPendingRow ? (
          <ChatMessage
            from="agent"
            avatar={
              <AgentStatusAvatar
                status={agentStatus}
                name={AUTOPILOT_NAME}
                src={AUTOPILOT_AVATAR_URL}
                size="sm"
              />
            }
          >
            <Text variant="body" as="span" tone="muted">
              {agentActivity}…
            </Text>
          </ChatMessage>
        ) : null}
        {reviewTarget?.kind === "graph" ? (
          <CopilotPendingReviews
            graphExecId={reviewTarget.graphExecId}
            graphId={reviewTarget.graphId}
          />
        ) : null}
        {sessionId ? (
          <CopilotPendingReviews
            chatSessionId={sessionId}
            pollWhileEmpty={reviewTarget?.kind === "chat"}
            refetchKey={countHeldCalls(messages)}
          />
        ) : null}
        {queuedMessages.map((text, index) => (
          <ChatMessage
            key={`queued-${index}`}
            from="user"
            className="opacity-60"
          >
            {text}
          </ChatMessage>
        ))}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
