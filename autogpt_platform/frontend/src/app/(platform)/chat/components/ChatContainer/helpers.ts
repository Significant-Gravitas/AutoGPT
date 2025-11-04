import type { ChatMessageData } from "@/components/molecules/ChatMessage/useChatMessage";
import type { ToolResult } from "@/types/chat";

/**
 * Type guard to validate message structure from backend.
 *
 * @param msg - The message to validate
 * @returns True if the message has valid structure
 */
export function isValidMessage(
  msg: unknown
): msg is Record<string, unknown> {
  if (typeof msg !== "object" || msg === null) {
    return false;
  }

  const m = msg as Record<string, unknown>;

  // Validate required fields
  if (typeof m.role !== "string") {
    return false;
  }

  // Content can be string or undefined
  if (m.content !== undefined && typeof m.content !== "string") {
    return false;
  }

  return true;
}

/**
 * Type guard to validate tool_calls array structure.
 *
 * @param value - The value to validate
 * @returns True if value is a valid tool_calls array
 */
export function isToolCallArray(
  value: unknown
): value is Array<{
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
      typeof item.function.arguments === "string"
  );
}

/**
 * Type guard to validate agent data structure.
 *
 * @param value - The value to validate
 * @returns True if value is a valid agents array
 */
export function isAgentArray(
  value: unknown
): value is Array<{
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
      (!("version" in item) || typeof item.version === "number")
  );
}

/**
 * Extracts a JSON object embedded within an error message string.
 *
 * This handles the edge case where the backend returns error messages
 * containing JSON objects with credential requirements or other structured data.
 * Uses manual brace matching to extract the first balanced JSON object.
 *
 * @param message - The error message that may contain embedded JSON
 * @returns The parsed JSON object, or null if no valid JSON found
 *
 * @example
 * ```ts
 * const msg = "Error: Missing credentials {\"missing_credentials\": {...}}";
 * const result = extractJsonFromErrorMessage(msg);
 * // Returns: { missing_credentials: {...} }
 * ```
 */
export function extractJsonFromErrorMessage(
  message: string,
): Record<string, unknown> | null {
  try {
    const start = message.indexOf("{");
    if (start === -1) {
      return null;
    }

    // Extract first balanced JSON object using brace matching
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

/**
 * Parses a tool result and converts it to the appropriate ChatMessageData type.
 *
 * Handles specialized tool response types like:
 * - no_results: Search returned no matches
 * - agent_carousel: List of agents to display
 * - execution_started: Agent execution began
 * - Generic tool responses: Raw tool output
 *
 * @param result - The tool result to parse (may be string or object)
 * @param toolId - The unique identifier for this tool call
 * @param toolName - The name of the tool that was called
 * @param timestamp - Optional timestamp for the response
 * @returns The appropriate ChatMessageData object, or null for setup_requirements
 */
export function parseToolResponse(
  result: ToolResult,
  toolId: string,
  toolName: string,
  timestamp?: Date,
): ChatMessageData | null {
  // Try to parse as JSON if it's a string
  let parsedResult: Record<string, unknown> | null = null;

  try {
    parsedResult =
      typeof result === "string"
        ? JSON.parse(result)
        : (result as Record<string, unknown>);
  } catch {
    // If parsing fails, we'll use the generic tool response
    parsedResult = null;
  }

  // Handle structured response types
  if (parsedResult && typeof parsedResult === "object") {
    const responseType = parsedResult.type as string | undefined;

    // Handle no_results response
    if (responseType === "no_results") {
      return {
        type: "no_results",
        message: (parsedResult.message as string) || "No results found",
        suggestions: (parsedResult.suggestions as string[]) || [],
        sessionId: parsedResult.session_id as string | undefined,
        timestamp: timestamp || new Date(),
      };
    }

    // Handle agent_carousel response
    if (responseType === "agent_carousel") {
      const agentsData = parsedResult.agents;

      // Validate agents array structure before using it
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

    // Handle execution_started response
    if (responseType === "execution_started") {
      return {
        type: "execution_started",
        executionId: (parsedResult.execution_id as string) || "",
        agentName: parsedResult.agent_name as string | undefined,
        message: parsedResult.message as string | undefined,
        timestamp: timestamp || new Date(),
      };
    }

    // Handle setup_requirements - return null so caller can handle it specially
    if (responseType === "setup_requirements") {
      return null;
    }
  }

  // Generic tool response
  return {
    type: "tool_response",
    toolId,
    toolName,
    result,
    success: true,
    timestamp: timestamp || new Date(),
  };
}

/**
 * Type guard to validate user readiness structure from backend.
 *
 * @param value - The value to validate
 * @returns True if the value matches the UserReadiness structure
 */
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

/**
 * Type guard to validate missing credentials structure.
 *
 * @param value - The value to validate
 * @returns True if the value is a valid missing credentials record
 */
export function isMissingCredentials(
  value: unknown,
): value is Record<string, Record<string, unknown>> {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  // Check that all values are objects
  return Object.values(value).every(
    (v) => typeof v === "object" && v !== null,
  );
}

/**
 * Type guard to validate setup info structure.
 *
 * @param value - The value to validate
 * @returns True if the value contains valid setup info
 */
export function isSetupInfo(
  value: unknown,
): value is {
  user_readiness?: Record<string, unknown>;
  agent_name?: string;
} {
  return (
    typeof value === "object" &&
    value !== null &&
    (!("user_readiness" in value) ||
      typeof (value as any).user_readiness === "object") &&
    (!("agent_name" in value) ||
      typeof (value as any).agent_name === "string")
  );
}

/**
 * Extract credentials requirements from setup info result.
 *
 * Used when a tool response indicates missing credentials are needed
 * to execute an agent.
 *
 * @param parsedResult - The parsed tool response result
 * @returns ChatMessageData for credentials_needed, or null if no credentials needed
 */
export function extractCredentialsNeeded(
  parsedResult: Record<string, unknown>
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

    // If there are missing credentials, create the message with ALL credentials
    if (missingCreds && Object.keys(missingCreds).length > 0) {
      const agentName = (setupInfo?.agent_name as string) || "this agent";

      // Map all missing credentials to the array format
      const credentials = Object.values(missingCreds).map((credInfo) => ({
        provider: (credInfo.provider as string) || "unknown",
        providerName:
          (credInfo.provider_name as string) ||
          (credInfo.provider as string) ||
          "Unknown Provider",
        credentialType: (credInfo.type as "api_key" | "oauth2" | "user_password" | "host_scoped") || "api_key",
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
