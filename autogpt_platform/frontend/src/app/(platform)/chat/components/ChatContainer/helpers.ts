import type { ChatMessageData } from "@/app/(platform)/chat/components/ChatMessage/useChatMessage";
import type { ToolResult } from "@/types/chat";

export function createUserMessage(content: string): ChatMessageData {
  return {
    type: "message",
    role: "user",
    content,
    timestamp: new Date(),
  };
}

export function filterAuthMessages(
  messages: ChatMessageData[],
): ChatMessageData[] {
  return messages.filter(
    (msg) => msg.type !== "credentials_needed" && msg.type !== "login_needed",
  );
}

export function isValidMessage(msg: unknown): msg is Record<string, unknown> {
  if (typeof msg !== "object" || msg === null) {
    return false;
  }
  const m = msg as Record<string, unknown>;
  if (typeof m.role !== "string") {
    return false;
  }
  if (m.content !== undefined && typeof m.content !== "string") {
    return false;
  }
  return true;
}

export function isToolCallArray(value: unknown): value is Array<{
  id: string;
  type: string;
  function: { name: string; arguments: string };
}> {
  if (!Array.isArray(value)) {
    return false;
  }
  return value.every(
    (item) =>
      typeof item === "object" &&
      item !== null &&
      "id" in item &&
      typeof item.id === "string" &&
      "type" in item &&
      typeof item.type === "string" &&
      "function" in item &&
      typeof item.function === "object" &&
      item.function !== null &&
      "name" in item.function &&
      typeof item.function.name === "string" &&
      "arguments" in item.function &&
      typeof item.function.arguments === "string",
  );
}

export function isAgentArray(value: unknown): value is Array<{
  id: string;
  name: string;
  description: string;
  version?: number;
}> {
  if (!Array.isArray(value)) {
    return false;
  }
  return value.every(
    (item) =>
      typeof item === "object" &&
      item !== null &&
      "id" in item &&
      typeof item.id === "string" &&
      "name" in item &&
      typeof item.name === "string" &&
      "description" in item &&
      typeof item.description === "string" &&
      (!("version" in item) || typeof item.version === "number"),
  );
}

export function extractJsonFromErrorMessage(
  message: string,
): Record<string, unknown> | null {
  try {
    const start = message.indexOf("{");
    if (start === -1) {
      return null;
    }
    let depth = 0;
    let end = -1;
    for (let i = start; i < message.length; i++) {
      const ch = message[i];
      if (ch === "{") {
        depth++;
      } else if (ch === "}") {
        depth--;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) {
      return null;
    }
    const jsonStr = message.slice(start, end + 1);
    return JSON.parse(jsonStr) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function parseToolResponse(
  result: ToolResult,
  toolId: string,
  toolName: string,
  timestamp?: Date,
): ChatMessageData | null {
  let parsedResult: Record<string, unknown> | null = null;
  try {
    parsedResult =
      typeof result === "string"
        ? JSON.parse(result)
        : (result as Record<string, unknown>);
  } catch {
    parsedResult = null;
  }
  if (parsedResult && typeof parsedResult === "object") {
    const responseType = parsedResult.type as string | undefined;
    if (responseType === "no_results") {
      return {
        type: "tool_response",
        toolId,
        toolName,
        result: (parsedResult.message as string) || "No results found",
        success: true,
        timestamp: timestamp || new Date(),
      };
    }
    if (responseType === "agent_carousel") {
      const agentsData = parsedResult.agents;
      if (isAgentArray(agentsData)) {
        return {
          type: "agent_carousel",
          agents: agentsData,
          totalCount: parsedResult.total_count as number | undefined,
          timestamp: timestamp || new Date(),
        };
      } else {
        console.warn("Invalid agents array in agent_carousel response");
      }
    }
    if (responseType === "execution_started") {
      return {
        type: "execution_started",
        executionId: (parsedResult.execution_id as string) || "",
        agentName: parsedResult.agent_name as string | undefined,
        message: parsedResult.message as string | undefined,
        timestamp: timestamp || new Date(),
      };
    }
    if (responseType === "need_login") {
      return {
        type: "login_needed",
        message:
          (parsedResult.message as string) ||
          "Please sign in to use chat and agent features",
        sessionId: (parsedResult.session_id as string) || "",
        agentInfo: parsedResult.agent_info as
          | {
              graph_id: string;
              name: string;
              trigger_type: string;
            }
          | undefined,
        timestamp: timestamp || new Date(),
      };
    }
    if (responseType === "setup_requirements") {
      return null;
    }
  }
  return {
    type: "tool_response",
    toolId,
    toolName,
    result,
    success: true,
    timestamp: timestamp || new Date(),
  };
}

export function isUserReadiness(
  value: unknown,
): value is { missing_credentials?: Record<string, unknown> } {
  return (
    typeof value === "object" &&
    value !== null &&
    (!("missing_credentials" in value) ||
      typeof (value as any).missing_credentials === "object")
  );
}

export function isMissingCredentials(
  value: unknown,
): value is Record<string, Record<string, unknown>> {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return Object.values(value).every((v) => typeof v === "object" && v !== null);
}

export function isSetupInfo(value: unknown): value is {
  user_readiness?: Record<string, unknown>;
  agent_name?: string;
} {
  return (
    typeof value === "object" &&
    value !== null &&
    (!("user_readiness" in value) ||
      typeof (value as any).user_readiness === "object") &&
    (!("agent_name" in value) || typeof (value as any).agent_name === "string")
  );
}

export function extractCredentialsNeeded(
  parsedResult: Record<string, unknown>,
): ChatMessageData | null {
  try {
    const setupInfo = parsedResult?.setup_info as
      | Record<string, unknown>
      | undefined;
    const userReadiness = setupInfo?.user_readiness as
      | Record<string, unknown>
      | undefined;
    const missingCreds = userReadiness?.missing_credentials as
      | Record<string, Record<string, unknown>>
      | undefined;
    if (missingCreds && Object.keys(missingCreds).length > 0) {
      const agentName = (setupInfo?.agent_name as string) || "this agent";
      const credentials = Object.values(missingCreds).map((credInfo) => ({
        provider: (credInfo.provider as string) || "unknown",
        providerName:
          (credInfo.provider_name as string) ||
          (credInfo.provider as string) ||
          "Unknown Provider",
        credentialType:
          (credInfo.type as
            | "api_key"
            | "oauth2"
            | "user_password"
            | "host_scoped") || "api_key",
        title:
          (credInfo.title as string) ||
          `${(credInfo.provider_name as string) || (credInfo.provider as string)} credentials`,
        scopes: credInfo.scopes as string[] | undefined,
      }));
      return {
        type: "credentials_needed",
        credentials,
        message: `To run ${agentName}, you need to add ${credentials.length === 1 ? "credentials" : `${credentials.length} credentials`}.`,
        agentName,
        timestamp: new Date(),
      };
    }
    return null;
  } catch (err) {
    console.error("Failed to extract credentials from setup info:", err);
    return null;
  }
}
