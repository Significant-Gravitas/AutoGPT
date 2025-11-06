import { formatDistanceToNow } from "date-fns";
import type { ToolArguments, ToolResult } from "@/types/chat";

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
      credentials: Array<{
        provider: string;
        providerName: string;
        credentialType: "api_key" | "oauth2" | "user_password" | "host_scoped";
        title: string;
        scopes?: string[];
      }>;
      message: string;
      agentName?: string;
      timestamp?: string | Date;
    }
  | {
      type: "no_results";
      message: string;
      suggestions?: string[];
      sessionId?: string;
      timestamp?: string | Date;
    }
  | {
      type: "agent_carousel";
      agents: Array<{
        id: string;
        name: string;
        description: string;
        version?: number;
      }>;
      totalCount?: number;
      timestamp?: string | Date;
    }
  | {
      type: "execution_started";
      executionId: string;
      agentName?: string;
      message?: string;
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
  };
}
