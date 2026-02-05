import type { ToolArguments, ToolResult } from "@/types/chat";
import { formatDistanceToNow } from "date-fns";

export type ChatMessageData =
  | {
      type: "message";
      role: "user" | "assistant" | "system";
      content: string;
      timestamp?: string | Date;
    }
  | {
      type: "tool_call";
      toolId: string;
      toolName: string;
      arguments?: ToolArguments;
      timestamp?: string | Date;
    }
  | {
      type: "tool_response";
      toolId: string;
      toolName: string;
      result: ToolResult;
      success?: boolean;
      timestamp?: string | Date;
    }
  | {
      type: "login_needed";
      toolName: string;
      message: string;
      sessionId: string;
      agentInfo?: {
        graph_id: string;
        name: string;
        trigger_type: string;
      };
      timestamp?: string | Date;
    }
  | {
      type: "credentials_needed";
      toolName: string;
      credentials: Array<{
        provider: string;
        providerName: string;
        credentialTypes: Array<
          "api_key" | "oauth2" | "user_password" | "host_scoped"
        >;
        title: string;
        scopes?: string[];
      }>;
      message: string;
      agentName?: string;
      timestamp?: string | Date;
    }
  | {
      type: "no_results";
      toolName: string;
      message: string;
      suggestions?: string[];
      sessionId?: string;
      timestamp?: string | Date;
    }
  | {
      type: "agent_carousel";
      toolId: string;
      toolName: string;
      agents: Array<{
        id: string;
        name: string;
        description: string;
        version?: number;
        image_url?: string;
      }>;
      totalCount?: number;
      timestamp?: string | Date;
    }
  | {
      type: "execution_started";
      toolId: string;
      toolName: string;
      executionId: string;
      agentName?: string;
      message?: string;
      libraryAgentLink?: string;
      timestamp?: string | Date;
    }
  | {
      type: "inputs_needed";
      toolName: string;
      agentName?: string;
      agentId?: string;
      graphVersion?: number;
      inputSchema: Record<string, any>;
      credentialsSchema?: Record<string, any>;
      message: string;
      timestamp?: string | Date;
    }
  | {
      type: "clarification_needed";
      toolName: string;
      questions: Array<{
        question: string;
        keyword: string;
        example?: string;
      }>;
      message: string;
      sessionId: string;
      timestamp?: string | Date;
    }
  | {
      type: "operation_started";
      toolName: string;
      toolId: string;
      operationId: string;
      taskId?: string; // For SSE reconnection
      message: string;
      timestamp?: string | Date;
    }
  | {
      type: "operation_pending";
      toolName: string;
      toolId: string;
      operationId: string;
      message: string;
      timestamp?: string | Date;
    }
  | {
      type: "operation_in_progress";
      toolName: string;
      toolCallId: string;
      message: string;
      timestamp?: string | Date;
    };

export function useChatMessage(message: ChatMessageData) {
  const formattedTimestamp = message.timestamp
    ? formatDistanceToNow(new Date(message.timestamp), { addSuffix: true })
    : "Just now";

  return {
    formattedTimestamp,
    isUser: message.type === "message" && message.role === "user",
    isAssistant: message.type === "message" && message.role === "assistant",
    isSystem: message.type === "message" && message.role === "system",
    isToolCall: message.type === "tool_call",
    isToolResponse: message.type === "tool_response",
    isLoginNeeded: message.type === "login_needed",
    isCredentialsNeeded: message.type === "credentials_needed",
    isNoResults: message.type === "no_results",
    isAgentCarousel: message.type === "agent_carousel",
    isExecutionStarted: message.type === "execution_started",
    isInputsNeeded: message.type === "inputs_needed",
    isClarificationNeeded: message.type === "clarification_needed",
    isOperationStarted: message.type === "operation_started",
    isOperationPending: message.type === "operation_pending",
    isOperationInProgress: message.type === "operation_in_progress",
  };
}
