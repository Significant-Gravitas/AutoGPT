"use client";

import { cn } from "@/lib/utils";
import { RobotIcon, UserIcon, CheckCircleIcon } from "@phosphor-icons/react";
import { useCallback } from "react";
import { MessageBubble } from "@/app/(platform)/chat/components/MessageBubble/MessageBubble";
import { MarkdownContent } from "@/app/(platform)/chat/components/MarkdownContent/MarkdownContent";
import { ToolCallMessage } from "@/app/(platform)/chat/components/ToolCallMessage/ToolCallMessage";
import { ToolResponseMessage } from "@/app/(platform)/chat/components/ToolResponseMessage/ToolResponseMessage";
import { AuthPromptWidget } from "@/app/(platform)/chat/components/AuthPromptWidget/AuthPromptWidget";
import { ChatCredentialsSetup } from "@/app/(platform)/chat/components/ChatCredentialsSetup/ChatCredentialsSetup";
import { NoResultsMessage } from "@/app/(platform)/chat/components/NoResultsMessage/NoResultsMessage";
import { AgentCarouselMessage } from "@/app/(platform)/chat/components/AgentCarouselMessage/AgentCarouselMessage";
import { ExecutionStartedMessage } from "@/app/(platform)/chat/components/ExecutionStartedMessage/ExecutionStartedMessage";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useChatMessage, type ChatMessageData } from "./useChatMessage";

export interface ChatMessageProps {
  message: ChatMessageData;
  className?: string;
  onDismissLogin?: () => void;
  onDismissCredentials?: () => void;
  onSendMessage?: (content: string, isUserMessage?: boolean) => void;
}

export function ChatMessage({
  message,
  className,
  onDismissCredentials,
  onSendMessage,
}: ChatMessageProps) {
  const { user } = useSupabase();
  const {
    formattedTimestamp,
    isUser,
    isAssistant,
    isToolCall,
    isToolResponse,
    isLoginNeeded,
    isCredentialsNeeded,
    isNoResults,
    isAgentCarousel,
    isExecutionStarted,
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
          <div className="my-4 overflow-hidden rounded-lg border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 dark:border-green-800 dark:from-green-950/30 dark:to-emerald-950/30">
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
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Successfully Authenticated
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
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
          returnUrl="/chat"
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

  // Render tool response messages
  if (isToolResponse && message.type === "tool_response") {
    return (
      <div className={cn("px-4 py-2", className)}>
        <ToolResponseMessage
          toolId={message.toolId}
          toolName={message.toolName}
          result={message.result}
          success={message.success}
        />
      </div>
    );
  }

  // Render no results messages
  if (isNoResults && message.type === "no_results") {
    return (
      <NoResultsMessage
        message={message.message}
        suggestions={message.suggestions}
        className={className}
      />
    );
  }

  // Render agent carousel messages
  if (isAgentCarousel && message.type === "agent_carousel") {
    return (
      <AgentCarouselMessage
        agents={message.agents}
        totalCount={message.totalCount}
        className={className}
      />
    );
  }

  // Render execution started messages
  if (isExecutionStarted && message.type === "execution_started") {
    return (
      <ExecutionStartedMessage
        executionId={message.executionId}
        agentName={message.agentName}
        message={message.message}
        className={className}
      />
    );
  }

  // Render regular chat messages
  if (message.type === "message") {
    return (
      <div
        className={cn(
          "flex gap-3 px-4 py-4",
          isUser && "flex-row-reverse",
          className,
        )}
      >
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-full",
              isUser && "bg-zinc-200 dark:bg-zinc-700",
              isAssistant && "bg-purple-600 dark:bg-purple-500",
            )}
          >
            {isUser ? (
              <UserIcon className="h-5 w-5 text-zinc-700 dark:text-zinc-200" />
            ) : (
              <RobotIcon className="h-5 w-5 text-white" />
            )}
          </div>
        </div>

        {/* Message Content */}
        <div className={cn("flex max-w-[70%] flex-col", isUser && "items-end")}>
          <MessageBubble variant={isUser ? "user" : "assistant"}>
            <MarkdownContent content={message.content} />
          </MessageBubble>

          {/* Timestamp */}
          <span
            className={cn(
              "mt-1 text-xs text-zinc-500 dark:text-zinc-400",
              isUser && "text-right",
            )}
          >
            {formattedTimestamp}
          </span>
        </div>
      </div>
    );
  }

  // Fallback for unknown message types
  return null;
}
