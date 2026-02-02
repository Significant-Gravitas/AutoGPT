import { toast } from "sonner";
import type { StreamChunk } from "../../chat-types";
import type { HandlerDependencies } from "./handlers";
import {
  handleError,
  handleLoginNeeded,
  handleStreamEnd,
  handleTextChunk,
  handleTextEnded,
  handleToolCallStart,
  handleToolResponse,
  isRegionBlockedError,
} from "./handlers";

export function createStreamEventDispatcher(
  deps: HandlerDependencies,
): (chunk: StreamChunk) => void {
  return function dispatchStreamEvent(chunk: StreamChunk): void {
    if (
      chunk.type === "text_chunk" ||
      chunk.type === "tool_call_start" ||
      chunk.type === "tool_response" ||
      chunk.type === "login_needed" ||
      chunk.type === "need_login" ||
      chunk.type === "error"
    ) {
      if (!deps.hasResponseRef.current) {
        console.info("[ChatStream] First response chunk:", {
          type: chunk.type,
          sessionId: deps.sessionId,
        });
      }
      deps.hasResponseRef.current = true;
    }

    switch (chunk.type) {
      case "stream_start":
        // Store task ID for SSE reconnection
        if (chunk.taskId && deps.onActiveTaskStarted) {
          console.info("[ChatStream] Stream started with task ID:", {
            sessionId: deps.sessionId,
            taskId: chunk.taskId,
          });
          deps.onActiveTaskStarted({
            taskId: chunk.taskId,
            operationId: chunk.taskId, // Use taskId as operationId for chat streams
            toolName: "chat",
            toolCallId: "chat_stream",
          });
        }
        break;

      case "text_chunk":
        handleTextChunk(chunk, deps);
        break;

      case "text_ended":
        handleTextEnded(chunk, deps);
        break;

      case "tool_call_start":
        handleToolCallStart(chunk, deps);
        break;

      case "tool_response":
        handleToolResponse(chunk, deps);
        break;

      case "login_needed":
      case "need_login":
        handleLoginNeeded(chunk, deps);
        break;

      case "stream_end":
        // Note: "finish" type from backend gets normalized to "stream_end" by normalizeStreamChunk
        console.info("[SSE-RECONNECT] Stream ended:", {
          sessionId: deps.sessionId,
          hasResponse: deps.hasResponseRef.current,
          chunkCount: deps.streamingChunksRef.current.length,
        });
        handleStreamEnd(chunk, deps);
        break;

      case "error":
        const isRegionBlocked = isRegionBlockedError(chunk);
        handleError(chunk, deps);
        // Show toast at dispatcher level to avoid circular dependencies
        if (!isRegionBlocked) {
          toast.error("Chat Error", {
            description: chunk.message || chunk.content || "An error occurred",
          });
        }
        break;

      case "usage":
        // TODO: Handle usage for display
        break;

      default:
        console.warn("Unknown stream chunk type:", chunk);
    }
  };
}
