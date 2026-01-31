"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { cn } from "@/lib/utils";
import {
  ArrowsClockwiseIcon,
  CheckCircleIcon,
  CheckIcon,
} from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { AgentCarouselMessage } from "../AgentCarouselMessage/AgentCarouselMessage";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { AuthPromptWidget } from "../AuthPromptWidget/AuthPromptWidget";
import { ChatCredentialsSetup } from "../ChatCredentialsSetup/ChatCredentialsSetup";
import { ClarificationQuestionsWidget } from "../ClarificationQuestionsWidget/ClarificationQuestionsWidget";
import { ExecutionStartedMessage } from "../ExecutionStartedMessage/ExecutionStartedMessage";
import { PendingOperationWidget } from "../PendingOperationWidget/PendingOperationWidget";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { NoResultsMessage } from "../NoResultsMessage/NoResultsMessage";
import { ToolCallMessage } from "../ToolCallMessage/ToolCallMessage";
import { ToolResponseMessage } from "../ToolResponseMessage/ToolResponseMessage";
import { UserChatBubble } from "../UserChatBubble/UserChatBubble";
import { useChatMessage, type ChatMessageData } from "./useChatMessage";

function stripInternalReasoning(content: string): string {
  const cleaned = content.replace(
    /<internal_reasoning>[\s\S]*?<\/internal_reasoning>/gi,
    "",
  );
  return cleaned.replace(/\n{3,}/g, "\n\n").trim();
}

function getDisplayContent(message: ChatMessageData, isUser: boolean): string {
  if (message.type !== "message") return "";
  if (isUser) return message.content;
  return stripInternalReasoning(message.content);
}

export interface ChatMessageProps {
  message: ChatMessageData;
  messages?: ChatMessageData[];
  index?: number;
  isStreaming?: boolean;
  className?: string;
  onDismissLogin?: () => void;
  onDismissCredentials?: () => void;
  onSendMessage?: (content: string, isUserMessage?: boolean) => void;
  agentOutput?: ChatMessageData;
  isFinalMessage?: boolean;
}

export function ChatMessage({
  message,
  messages = [],
  index = -1,
  isStreaming = false,
  className,
  onDismissCredentials,
  onSendMessage,
  agentOutput,
  isFinalMessage = true,
}: ChatMessageProps) {
  const { user } = useSupabase();
  const router = useRouter();
  const [copied, setCopied] = useState(false);
  const {
    isUser,
    isToolCall,
    isToolResponse,
    isLoginNeeded,
    isCredentialsNeeded,
    isClarificationNeeded,
    isOperationStarted,
    isOperationPending,
    isOperationInProgress,
  } = useChatMessage(message);
  const displayContent = getDisplayContent(message, isUser);

  const handleAllCredentialsComplete = useCallback(
    function handleAllCredentialsComplete() {
      // Send a user message that explicitly asks to retry the setup
      // This ensures the LLM calls get_required_setup_info again and proceeds with execution
      if (onSendMessage) {
        onSendMessage(
          "I've configured the required credentials. Please check if everything is ready and proceed with setting up the agent.",
        );
      }
      // Optionally dismiss the credentials prompt
      if (onDismissCredentials) {
        onDismissCredentials();
      }
    },
    [onSendMessage, onDismissCredentials],
  );

  function handleCancelCredentials() {
    // Dismiss the credentials prompt
    if (onDismissCredentials) {
      onDismissCredentials();
    }
  }

  function handleClarificationAnswers(answers: Record<string, string>) {
    if (onSendMessage) {
      const contextMessage = Object.entries(answers)
        .map(([keyword, answer]) => `${keyword}: ${answer}`)
        .join("\n");

      onSendMessage(
        `I have the answers to your questions:\n\n${contextMessage}\n\nPlease proceed with creating the agent.`,
      );
    }
  }

  const handleCopy = useCallback(
    async function handleCopy() {
      if (message.type !== "message") return;
      if (!displayContent) return;

      try {
        await navigator.clipboard.writeText(displayContent);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error("Failed to copy:", error);
      }
    },
    [displayContent, message],
  );

  const handleTryAgain = useCallback(() => {
    if (message.type !== "message" || !onSendMessage) return;
    onSendMessage(message.content, message.role === "user");
  }, [message, onSendMessage]);

  const handleViewExecution = useCallback(() => {
    if (message.type === "execution_started" && message.libraryAgentLink) {
      router.push(message.libraryAgentLink);
    }
  }, [message, router]);

  // Render credentials needed messages
  if (isCredentialsNeeded && message.type === "credentials_needed") {
    return (
      <ChatCredentialsSetup
        credentials={message.credentials}
        agentName={message.agentName}
        message={message.message}
        onAllCredentialsComplete={handleAllCredentialsComplete}
        onCancel={handleCancelCredentials}
        className={className}
      />
    );
  }

  if (isClarificationNeeded && message.type === "clarification_needed") {
    const hasUserReplyAfter =
      index >= 0 &&
      messages
        .slice(index + 1)
        .some((m) => m.type === "message" && m.role === "user");

    return (
      <ClarificationQuestionsWidget
        questions={message.questions}
        message={message.message}
        sessionId={message.sessionId}
        onSubmitAnswers={handleClarificationAnswers}
        isAnswered={hasUserReplyAfter}
        className={className}
      />
    );
  }

  // Render login needed messages
  if (isLoginNeeded && message.type === "login_needed") {
    // If user is already logged in, show success message instead of auth prompt
    if (user) {
      return (
        <div className={cn("px-4 py-2", className)}>
          <div className="my-4 overflow-hidden rounded-lg border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
            <div className="px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-600">
                  <CheckCircleIcon
                    size={20}
                    weight="fill"
                    className="text-white"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-neutral-900">
                    Successfully Authenticated
                  </h3>
                  <p className="text-sm text-neutral-600">
                    You&apos;re now signed in and ready to continue
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // Show auth prompt if not logged in
    return (
      <div className={cn("px-4 py-2", className)}>
        <AuthPromptWidget
          message={message.message}
          sessionId={message.sessionId}
          agentInfo={message.agentInfo}
        />
      </div>
    );
  }

  // Render tool call messages
  if (isToolCall && message.type === "tool_call") {
    // Check if this tool call is currently streaming
    // A tool call is streaming if:
    // 1. isStreaming is true
    // 2. This is the last tool_call message
    // 3. There's no tool_response for this tool call yet
    const isToolCallStreaming =
      isStreaming &&
      index >= 0 &&
      (() => {
        // Find the last tool_call index
        let lastToolCallIndex = -1;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].type === "tool_call") {
            lastToolCallIndex = i;
            break;
          }
        }
        // Check if this is the last tool_call and there's no response yet
        if (index === lastToolCallIndex) {
          // Check if there's a tool_response for this tool call
          const hasResponse = messages
            .slice(index + 1)
            .some(
              (msg) =>
                msg.type === "tool_response" && msg.toolId === message.toolId,
            );
          return !hasResponse;
        }
        return false;
      })();

    return (
      <div className={cn("px-4 py-2", className)}>
        <ToolCallMessage
          toolId={message.toolId}
          toolName={message.toolName}
          arguments={message.arguments}
          isStreaming={isToolCallStreaming}
        />
      </div>
    );
  }

  // Render no_results messages - use dedicated component, not ToolResponseMessage
  if (message.type === "no_results") {
    return (
      <div className={cn("px-4 py-2", className)}>
        <NoResultsMessage
          message={message.message}
          suggestions={message.suggestions}
        />
      </div>
    );
  }

  // Render agent_carousel messages - use dedicated component, not ToolResponseMessage
  if (message.type === "agent_carousel") {
    return (
      <div className={cn("px-4 py-2", className)}>
        <AgentCarouselMessage
          agents={message.agents}
          totalCount={message.totalCount}
        />
      </div>
    );
  }

  // Render execution_started messages - use dedicated component, not ToolResponseMessage
  if (message.type === "execution_started") {
    return (
      <div className={cn("px-4 py-2", className)}>
        <ExecutionStartedMessage
          executionId={message.executionId}
          agentName={message.agentName}
          message={message.message}
          onViewExecution={
            message.libraryAgentLink ? handleViewExecution : undefined
          }
        />
      </div>
    );
  }

  // Render operation_started messages (long-running background operations)
  if (isOperationStarted && message.type === "operation_started") {
    return (
      <PendingOperationWidget
        status="started"
        message={message.message}
        toolName={message.toolName}
        className={className}
      />
    );
  }

  // Render operation_pending messages (operations in progress when refreshing)
  if (isOperationPending && message.type === "operation_pending") {
    return (
      <PendingOperationWidget
        status="pending"
        message={message.message}
        toolName={message.toolName}
        className={className}
      />
    );
  }

  // Render operation_in_progress messages (duplicate request while operation running)
  if (isOperationInProgress && message.type === "operation_in_progress") {
    return (
      <PendingOperationWidget
        status="in_progress"
        message={message.message}
        toolName={message.toolName}
        className={className}
      />
    );
  }

  // Render tool response messages (but skip agent_output if it's being rendered inside assistant message)
  if (isToolResponse && message.type === "tool_response") {
    return (
      <div className={cn("px-4 py-2", className)}>
        <ToolResponseMessage
          toolId={message.toolId}
          toolName={message.toolName}
          result={message.result}
        />
      </div>
    );
  }

  // Render regular chat messages
  if (message.type === "message") {
    return (
      <div
        className={cn(
          "group relative flex w-full gap-3 px-4 py-3",
          isUser ? "justify-end" : "justify-start",
          className,
        )}
      >
        <div className="flex w-full max-w-3xl gap-3">
          <div
            className={cn(
              "flex min-w-0 flex-1 flex-col",
              isUser && "items-end",
            )}
          >
            {isUser ? (
              <UserChatBubble>
                <MarkdownContent content={displayContent} />
              </UserChatBubble>
            ) : (
              <AIChatBubble>
                <MarkdownContent content={displayContent} />
                {agentOutput && agentOutput.type === "tool_response" && (
                  <div className="mt-4">
                    <ToolResponseMessage
                      toolId={agentOutput.toolId}
                      toolName={agentOutput.toolName || "Agent Output"}
                      result={agentOutput.result}
                    />
                  </div>
                )}
              </AIChatBubble>
            )}
            <div
              className={cn(
                "flex gap-0",
                isUser ? "justify-end" : "justify-start",
              )}
            >
              {isUser && onSendMessage && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleTryAgain}
                  aria-label="Try again"
                >
                  <ArrowsClockwiseIcon className="size-4 text-zinc-600" />
                </Button>
              )}
              {!isUser && isFinalMessage && !isStreaming && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCopy}
                  aria-label="Copy message"
                  className="p-1"
                >
                  {copied ? (
                    <CheckIcon className="size-4 text-green-600" />
                  ) : (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="size-3 text-zinc-600"
                    >
                      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                    </svg>
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Fallback for unknown message types
  return null;
}
