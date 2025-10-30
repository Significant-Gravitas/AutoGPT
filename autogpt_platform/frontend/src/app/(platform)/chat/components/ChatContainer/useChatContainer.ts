import { useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import { useChatStream, type StreamChunk } from "@/hooks/useChatStream";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { ChatMessageData } from "@/components/molecules/ChatMessage/useChatMessage";

interface UseChatContainerArgs {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  onRefreshSession: () => Promise<void>;
}

interface UseChatContainerResult {
  messages: ChatMessageData[];
  streamingChunks: string[];
  isStreaming: boolean;
  error: Error | null;
  sendMessage: (content: string) => Promise<void>;
}

export function useChatContainer({
  sessionId,
  initialMessages,
  onRefreshSession,
}: UseChatContainerArgs): UseChatContainerResult {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
  const [hasTextChunks, setHasTextChunks] = useState(false);

  // Track streaming chunks in a ref so we can access the latest value in callbacks
  const streamingChunksRef = useRef<string[]>([]);

  // Track when tool calls were displayed to ensure minimum visibility time
  const toolCallTimestamps = useRef<Map<string, number>>(new Map());

  const { error, sendMessage: sendStreamMessage } = useChatStream();

  // Show streaming UI when we have text chunks, independent of connection state
  // This keeps the StreamingMessage visible during the transition to persisted message
  const isStreaming = hasTextChunks;

  // Convert initial messages to our format, filtering out empty messages
  const allMessages: ChatMessageData[] = [
    ...initialMessages
      .filter((msg: Record<string, unknown>) => {
        // Include messages with content OR tool_calls (tool_call messages have empty content)
        const content = String(msg.content || "").trim();
        const toolCalls = msg.tool_calls as unknown[] | undefined;
        return content.length > 0 || (toolCalls && toolCalls.length > 0);
      })
      .map((msg: Record<string, unknown>): ChatMessageData | null => {
        const content = String(msg.content || "");
        const role = String(msg.role || "assistant").toLowerCase();

        // Check if this is a tool_call message (assistant message with tool_calls)
        const toolCalls = msg.tool_calls as
          | Array<{
              id: string;
              type: string;
              function: { name: string; arguments: string };
            }>
          | undefined;

        if (role === "assistant" && toolCalls && toolCalls.length > 0) {
          // Skip tool_call messages from persisted history
          // We only show tool_calls during live streaming, not from history
          // The tool_response that follows it is what we want to display
          return null;
        }

        // Check if this is a tool response message (role="tool")
        if (role === "tool") {
          // Try to parse the content as JSON to detect structured responses
          try {
            const parsed = JSON.parse(content);
            if (parsed && typeof parsed === "object" && parsed.type) {
              // Handle no_results
              if (parsed.type === "no_results") {
                return {
                  type: "no_results",
                  message: parsed.message || "No results found",
                  suggestions: parsed.suggestions || [],
                  sessionId: parsed.session_id,
                  timestamp: msg.timestamp
                    ? new Date(msg.timestamp as string)
                    : undefined,
                };
              }

              // Handle agent_carousel
              if (
                parsed.type === "agent_carousel" &&
                Array.isArray(parsed.agents)
              ) {
                return {
                  type: "agent_carousel",
                  agents: parsed.agents,
                  totalCount: parsed.total_count,
                  timestamp: msg.timestamp
                    ? new Date(msg.timestamp as string)
                    : undefined,
                };
              }

              // Handle execution_started
              if (parsed.type === "execution_started") {
                return {
                  type: "execution_started",
                  executionId: parsed.execution_id || "",
                  agentName: parsed.agent_name,
                  message: parsed.message,
                  timestamp: msg.timestamp
                    ? new Date(msg.timestamp as string)
                    : undefined,
                };
              }
            }

            // Generic tool response - not a specialized type
            return {
              type: "tool_response",
              toolId: (msg.tool_call_id as string) || "",
              toolName: "unknown",
              result: parsed,
              success: true,
              timestamp: msg.timestamp
                ? new Date(msg.timestamp as string)
                : undefined,
            };
          } catch {
            // Not valid JSON, treat as string result
            return {
              type: "tool_response",
              toolId: (msg.tool_call_id as string) || "",
              toolName: "unknown",
              result: content,
              success: true,
              timestamp: msg.timestamp
                ? new Date(msg.timestamp as string)
                : undefined,
            };
          }
        }

        // Return as regular message
        return {
          type: "message",
          role: role as "user" | "assistant" | "system",
          content,
          timestamp: msg.timestamp
            ? new Date(msg.timestamp as string)
            : undefined,
        };
      })
      .filter((msg): msg is ChatMessageData => msg !== null), // Remove null entries
    ...messages,
  ];

  const sendMessage = useCallback(
    async function sendMessage(content: string) {
      if (!sessionId) {
        console.error("Cannot send message: no session ID");
        return;
      }

      // Add user message immediately
      const userMessage: ChatMessageData = {
        type: "message",
        role: "user",
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Clear streaming chunks and reset text flag
      setStreamingChunks([]);
      streamingChunksRef.current = [];
      setHasTextChunks(false);
      // Clear any pending tool call timestamps
      toolCallTimestamps.current.clear();

      try {
        // Stream the response
        await sendStreamMessage(
          sessionId,
          content,
          function handleChunk(chunk: StreamChunk) {
            if (chunk.type === "text_chunk" && chunk.content) {
              setHasTextChunks(true); // Mark that we have text chunks
              setStreamingChunks((prev) => {
                const updated = [...prev, chunk.content!];
                streamingChunksRef.current = updated;
                return updated;
              });
            } else if (chunk.type === "tool_call") {
              // Record the timestamp when this tool call was displayed
              toolCallTimestamps.current.set(chunk.tool_id!, Date.now());

              // Add tool call message
              const toolCallMessage: ChatMessageData = {
                type: "tool_call",
                toolId: chunk.tool_id!,
                toolName: chunk.tool_name!,
                arguments: chunk.arguments,
                timestamp: new Date(),
              };
              setMessages((prev) => [...prev, toolCallMessage]);
            } else if (chunk.type === "tool_response") {
              // Parse the tool response first so it's available for all checks
              let parsedResult: Record<string, unknown> | null = null;
              try {
                parsedResult =
                  typeof chunk.result === "string"
                    ? JSON.parse(chunk.result)
                    : (chunk.result as Record<string, unknown>);
              } catch {
                // If parsing fails, we'll use the generic tool response
                parsedResult = null;
              }

              // Ensure tool call was visible for at least 500ms
              const MIN_DISPLAY_TIME = 500; // milliseconds
              const toolCallTime = toolCallTimestamps.current.get(
                chunk.tool_id!,
              );

              const processToolResponse = async () => {
                if (toolCallTime) {
                  const elapsed = Date.now() - toolCallTime;
                  const remainingTime = MIN_DISPLAY_TIME - elapsed;

                  if (remainingTime > 0) {
                    await new Promise((resolve) =>
                      setTimeout(resolve, remainingTime),
                    );
                  }

                  // Clean up the timestamp
                  toolCallTimestamps.current.delete(chunk.tool_id!);
                }

                // Create the appropriate message based on response type
                let responseMessage: ChatMessageData;

                if (parsedResult && typeof parsedResult === "object") {
                  const responseType = parsedResult.type as string | undefined;

                  // Handle no_results response
                  if (responseType === "no_results") {
                    responseMessage = {
                      type: "no_results",
                      message:
                        (parsedResult.message as string) || "No results found",
                      suggestions: (parsedResult.suggestions as string[]) || [],
                      sessionId: parsedResult.session_id as string | undefined,
                      timestamp: new Date(),
                    };
                  } else if (responseType === "agent_carousel") {
                    // Handle agent_carousel response
                    const agentsData = parsedResult.agents as Array<{
                      id: string;
                      name: string;
                      description: string;
                      version?: number;
                    }>;
                    if (agentsData && Array.isArray(agentsData)) {
                      responseMessage = {
                        type: "agent_carousel",
                        agents: agentsData,
                        totalCount: parsedResult.total_count as
                          | number
                          | undefined,
                        timestamp: new Date(),
                      };
                    } else {
                      // Fallback to generic if agents array is invalid
                      responseMessage = {
                        type: "tool_response",
                        toolId: chunk.tool_id!,
                        toolName: chunk.tool_name!,
                        result: chunk.result!,
                        success: chunk.success,
                        timestamp: new Date(),
                      };
                    }
                  } else if (responseType === "execution_started") {
                    // Handle execution_started response
                    responseMessage = {
                      type: "execution_started",
                      executionId: (parsedResult.execution_id as string) || "",
                      agentName: parsedResult.agent_name as string | undefined,
                      message: parsedResult.message as string | undefined,
                      timestamp: new Date(),
                    };
                  } else {
                    // Generic tool response
                    responseMessage = {
                      type: "tool_response",
                      toolId: chunk.tool_id!,
                      toolName: chunk.tool_name!,
                      result: chunk.result!,
                      success: chunk.success,
                      timestamp: new Date(),
                    };
                  }
                } else {
                  // Generic tool response if parsing failed
                  responseMessage = {
                    type: "tool_response",
                    toolId: chunk.tool_id!,
                    toolName: chunk.tool_name!,
                    result: chunk.result!,
                    success: chunk.success,
                    timestamp: new Date(),
                  };
                }

                // Replace the tool_call message with the response message
                setMessages((prev) => {
                  // Find and replace the tool_call message with the same tool_id
                  const toolCallIndex = prev.findIndex(
                    (msg) =>
                      msg.type === "tool_call" && msg.toolId === chunk.tool_id,
                  );

                  if (toolCallIndex !== -1) {
                    // Replace the tool_call with the response
                    const newMessages = [...prev];
                    newMessages[toolCallIndex] = responseMessage;
                    return newMessages;
                  } else {
                    // Tool call not found (shouldn't happen), just append
                    return [...prev, responseMessage];
                  }
                });
              };

              // Process the tool response with potential delay
              processToolResponse();

              // Check if this is get_required_setup_info and has missing credentials
              if (
                chunk.tool_name === "get_required_setup_info" &&
                chunk.success &&
                parsedResult
              ) {
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

                  // If there are missing credentials, show a prompt
                  if (missingCreds && Object.keys(missingCreds).length > 0) {
                    // Get the first missing credential to show
                    const firstCredKey = Object.keys(missingCreds)[0];
                    const credInfo = missingCreds[firstCredKey];

                    const credentialsMessage: ChatMessageData = {
                      type: "credentials_needed",
                      provider: (credInfo.provider as string) || "unknown",
                      providerName:
                        (credInfo.provider_name as string) ||
                        (credInfo.provider as string) ||
                        "Unknown Provider",
                      credentialType: (credInfo.type as string) || "api_key",
                      title:
                        (credInfo.title as string) ||
                        (setupInfo?.agent_name as string) ||
                        "this agent",
                      message: `To run ${(setupInfo?.agent_name as string) || "this agent"}, you need to add ${(credInfo.provider_name as string) || (credInfo.provider as string)} credentials.`,
                      scopes: credInfo.scopes as string[] | undefined,
                      timestamp: new Date(),
                    };
                    setMessages((prev) => [...prev, credentialsMessage]);
                  }
                } catch (err) {
                  console.error(
                    "Failed to parse setup info for credentials check:",
                    err,
                  );
                  toast.error("Failed to parse credential requirements", {
                    description:
                      err instanceof Error
                        ? err.message
                        : "Could not process setup information",
                  });
                }
              }
            } else if (chunk.type === "login_needed") {
              // Add login needed message
              const loginNeededMessage: ChatMessageData = {
                type: "login_needed",
                message: chunk.message || "Authentication required to continue",
                sessionId: chunk.session_id || sessionId,
                timestamp: new Date(),
              };
              setMessages((prev) => [...prev, loginNeededMessage]);
            } else if (chunk.type === "stream_end") {
              // Convert streaming chunks into a completed assistant message
              // This provides seamless transition without flash or resize
              // Use ref to get the latest chunks value (not stale closure value)
              const completedContent = streamingChunksRef.current.join("");

              if (completedContent) {
                const assistantMessage: ChatMessageData = {
                  type: "message",
                  role: "assistant",
                  content: completedContent,
                  timestamp: new Date(),
                };
                setMessages((prev) => [...prev, assistantMessage]);
              }

              // Clear streaming state immediately now that we have the message
              setStreamingChunks([]);
              streamingChunksRef.current = [];
              setHasTextChunks(false);

              // Refresh session data in background, then clear local messages
              // The completed message from initialMessages will replace our local one
              onRefreshSession().then(() => {
                setMessages([]);
                toolCallTimestamps.current.clear();
              });
            } else if (chunk.type === "error") {
              const errorMessage =
                chunk.message || chunk.content || "An error occurred";
              console.error("Stream error:", errorMessage);
              toast.error("Chat Error", {
                description: errorMessage,
              });
            }
            // TODO: Handle usage for display
          },
        );
      } catch (err) {
        console.error("Failed to send message:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to send message";
        toast.error("Failed to send message", {
          description: errorMessage,
        });
      }
    },
    [sessionId, sendStreamMessage, onRefreshSession],
  );

  return {
    messages: allMessages,
    streamingChunks,
    isStreaming,
    error,
    sendMessage,
  };
}
