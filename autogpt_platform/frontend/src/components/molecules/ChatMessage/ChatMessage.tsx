import { cn } from "@/lib/utils";
import { Robot, User } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { MessageBubble } from "@/components/atoms/MessageBubble/MessageBubble";
import { ToolCallMessage } from "@/components/molecules/ToolCallMessage/ToolCallMessage";
import { ToolResponseMessage } from "@/components/molecules/ToolResponseMessage/ToolResponseMessage";
import { LoginPrompt } from "@/components/molecules/LoginPrompt/LoginPrompt";
import { CredentialsNeededPrompt } from "@/components/molecules/CredentialsNeededPrompt/CredentialsNeededPrompt";
import { NoResultsMessage } from "@/components/molecules/NoResultsMessage/NoResultsMessage";
import { AgentCarouselMessage } from "@/components/molecules/AgentCarouselMessage/AgentCarouselMessage";
import { ExecutionStartedMessage } from "@/components/molecules/ExecutionStartedMessage/ExecutionStartedMessage";
import { useChatMessage, type ChatMessageData } from "./useChatMessage";

export interface ChatMessageProps {
  message: ChatMessageData;
  className?: string;
  onDismissLogin?: () => void;
  onDismissCredentials?: () => void;
}

export function ChatMessage({
  message,
  className,
  onDismissLogin,
  onDismissCredentials,
}: ChatMessageProps) {
  const router = useRouter();
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

  function handleLogin() {
    // Save current path to return after login
    const currentPath = window.location.pathname + window.location.search;
    sessionStorage.setItem("post_login_redirect", currentPath);
    router.push("/login");
  }

  function handleContinueAsGuest() {
    // Dismiss the login prompt
    if (onDismissLogin) {
      onDismissLogin();
    }
  }

  function handleSetupCredentials() {
    // For now, redirect to integrations page
    // TODO: Deep link to specific provider when backend provides agent/block context
    router.push("/integrations");
  }

  function handleCancelCredentials() {
    // Dismiss the credentials prompt
    if (onDismissCredentials) {
      onDismissCredentials();
    }
  }

  // Render credentials needed messages
  if (isCredentialsNeeded && message.type === "credentials_needed") {
    return (
      <CredentialsNeededPrompt
        provider={message.provider}
        providerName={message.providerName}
        credentialType={message.credentialType}
        title={message.title}
        message={message.message}
        onSetupCredentials={handleSetupCredentials}
        onCancel={handleCancelCredentials}
        className={className}
      />
    );
  }

  // Render login needed messages
  if (isLoginNeeded && message.type === "login_needed") {
    return (
      <LoginPrompt
        message={message.message}
        onLogin={handleLogin}
        onContinueAsGuest={handleContinueAsGuest}
        className={className}
      />
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
              <User className="h-5 w-5 text-zinc-700 dark:text-zinc-200" />
            ) : (
              <Robot className="h-5 w-5 text-white" />
            )}
          </div>
        </div>

        {/* Message Content */}
        <div className={cn("flex max-w-[70%] flex-col", isUser && "items-end")}>
          <MessageBubble variant={isUser ? "user" : "assistant"}>
            <div className="whitespace-pre-wrap">{message.content}</div>
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
