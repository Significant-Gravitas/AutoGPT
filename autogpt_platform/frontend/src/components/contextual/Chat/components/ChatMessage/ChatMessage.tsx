"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { cn } from "@/lib/utils";
import {
  ArrowClockwise,
  CheckCircleIcon,
  CheckIcon,
  CopyIcon,
} from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { AgentCarouselMessage } from "../AgentCarouselMessage/AgentCarouselMessage";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { AuthPromptWidget } from "../AuthPromptWidget/AuthPromptWidget";
import { ChatCredentialsSetup } from "../ChatCredentialsSetup/ChatCredentialsSetup";
import { ExecutionStartedMessage } from "../ExecutionStartedMessage/ExecutionStartedMessage";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { NoResultsMessage } from "../NoResultsMessage/NoResultsMessage";
import { ToolCallMessage } from "../ToolCallMessage/ToolCallMessage";
import { ToolResponseMessage } from "../ToolResponseMessage/ToolResponseMessage";
import { UserChatBubble } from "../UserChatBubble/UserChatBubble";
import { useChatMessage, type ChatMessageData } from "./useChatMessage";
export interface ChatMessageProps {
  message: ChatMessageData;
  className?: string;
  onDismissLogin?: () => void;
  onDismissCredentials?: () => void;
  onSendMessage?: (content: string, isUserMessage?: boolean) => void;
  agentOutput?: ChatMessageData;
  isFinalMessage?: boolean;
}

export function ChatMessage({
  message,
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
  } = useChatMessage(message);

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

  const handleCopy = useCallback(async () => {
    if (message.type !== "message") return;

    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  }, [message]);

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
    return (
      <div className={cn("px-4 py-2", className)}>
        <ToolCallMessage
          toolId={message.toolId}
          toolName={message.toolName}
          arguments={message.arguments}
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
                <MarkdownContent content={message.content} />
              </UserChatBubble>
            ) : (
              <AIChatBubble>
                <MarkdownContent content={message.content} />
                {agentOutput &&
                  agentOutput.type === "tool_response" && (
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
                "flex gap-1",
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
                  <ArrowClockwise className="size-3 text-neutral-500" />
                </Button>
              )}
              {(isUser || isFinalMessage) && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCopy}
                  aria-label="Copy message"
                >
                  {copied ? (
                    <CheckIcon className="size-3 text-green-600" />
                  ) : (
                    <CopyIcon className="size-3 text-neutral-500" />
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
