import { INITIAL_MESSAGE_ID } from "./chat-constants";
import type {
  ActiveStream,
  StreamChunk,
  VercelStreamChunk,
} from "./chat-types";
import {
  INITIAL_RETRY_DELAY,
  MAX_RETRIES,
  normalizeStreamChunk,
  parseSSELine,
} from "./stream-utils";

function notifySubscribers(
  stream: ActiveStream,
  chunk: StreamChunk,
  skipStore = false,
) {
  if (!skipStore) {
    stream.chunks.push(chunk);
  }
  for (const callback of stream.onChunkCallbacks) {
    try {
      callback(chunk);
    } catch (err) {
      console.warn("[StreamExecutor] Subscriber callback error:", err);
    }
  }
}

/**
 * Options for stream execution.
 */
interface StreamExecutionOptions {
  /** The active stream state object */
  stream: ActiveStream;
  /** Execution mode: 'new' for new stream, 'reconnect' for task reconnection */
  mode: "new" | "reconnect";
  /** Message content (required for 'new' mode) */
  message?: string;
  /** Whether this is a user message (for 'new' mode) */
  isUserMessage?: boolean;
  /** Optional context for the message (for 'new' mode) */
  context?: { url: string; content: string };
  /** Task ID (required for 'reconnect' mode) */
  taskId?: string;
  /** Last message ID for replay (for 'reconnect' mode) */
  lastMessageId?: string;
  /** Current retry count (internal use) */
  retryCount?: number;
}

/**
 * Unified stream execution function that handles both new streams and task reconnection.
 *
 * For new streams:
 * - Posts a message to create a new chat stream
 * - Reads SSE chunks and notifies subscribers
 *
 * For reconnection:
 * - Connects to an existing task stream
 * - Replays messages from lastMessageId position
 * - Allows resumption of long-running operations
 */
async function executeStreamInternal(
  options: StreamExecutionOptions,
): Promise<void> {
  const {
    stream,
    mode,
    message,
    isUserMessage,
    context,
    taskId,
    lastMessageId = INITIAL_MESSAGE_ID,
    retryCount = 0,
  } = options;

  const { sessionId, abortController } = stream;
  const isReconnect = mode === "reconnect";
  const logPrefix = isReconnect ? "[SSE-RECONNECT]" : "[StreamExecutor]";

  if (isReconnect) {
    console.info(`${logPrefix} executeStream starting:`, {
      taskId,
      lastMessageId,
      retryCount,
    });
  }

  try {
    // Build URL and request options based on mode
    let url: string;
    let fetchOptions: RequestInit;

    if (isReconnect) {
      url = `/api/chat/tasks/${taskId}/stream?last_message_id=${encodeURIComponent(lastMessageId)}`;
      fetchOptions = {
        method: "GET",
        headers: {
          Accept: "text/event-stream",
        },
        signal: abortController.signal,
      };
      console.info(`${logPrefix} Fetching task stream:`, { url });
    } else {
      url = `/api/chat/sessions/${sessionId}/stream`;
      fetchOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message,
          is_user_message: isUserMessage,
          context: context || null,
        }),
        signal: abortController.signal,
      };
    }

    const response = await fetch(url, fetchOptions);

    if (isReconnect) {
      console.info(`${logPrefix} Task stream response:`, {
        status: response.status,
        ok: response.ok,
      });
    }

    if (!response.ok) {
      const errorText = await response.text();
      if (isReconnect) {
        console.error(`${logPrefix} Task stream error response:`, {
          status: response.status,
          errorText,
        });
      }
      // For reconnect: don't retry on 404/403 (permanent errors)
      const isPermanentError =
        isReconnect && (response.status === 404 || response.status === 403);
      const error = new Error(errorText || `HTTP ${response.status}`);
      (error as Error & { status?: number }).status = response.status;
      (error as Error & { isPermanent?: boolean }).isPermanent =
        isPermanentError;
      throw error;
    }

    if (!response.body) {
      throw new Error("Response body is null");
    }

    if (isReconnect) {
      console.info(`${logPrefix} Task stream connected, reading chunks...`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        if (isReconnect) {
          console.info(
            `${logPrefix} Task stream reader done (connection closed)`,
          );
        }
        notifySubscribers(stream, { type: "stream_end" });
        stream.status = "completed";
        return;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const data = parseSSELine(line);
        if (data !== null) {
          if (data === "[DONE]") {
            if (isReconnect) {
              console.info(`${logPrefix} Task stream received [DONE] signal`);
            }
            notifySubscribers(stream, { type: "stream_end" });
            stream.status = "completed";
            return;
          }

          try {
            const rawChunk = JSON.parse(data) as
              | StreamChunk
              | VercelStreamChunk;
            const chunk = normalizeStreamChunk(rawChunk);
            if (!chunk) continue;

            // Log first few chunks for debugging (reconnect mode only)
            if (isReconnect && stream.chunks.length < 3) {
              console.info(`${logPrefix} Task stream chunk received:`, {
                type: chunk.type,
                chunkIndex: stream.chunks.length,
              });
            }

            notifySubscribers(stream, chunk);

            if (chunk.type === "stream_end") {
              if (isReconnect) {
                console.info(
                  `${logPrefix} Task stream completed via stream_end chunk`,
                );
              }
              stream.status = "completed";
              return;
            }

            if (chunk.type === "error") {
              if (isReconnect) {
                console.error(`${logPrefix} Task stream error chunk:`, chunk);
              }
              stream.status = "error";
              stream.error = new Error(
                chunk.message || chunk.content || "Stream error",
              );
              return;
            }
          } catch (err) {
            console.warn(`${logPrefix} Failed to parse SSE chunk:`, err);
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      notifySubscribers(stream, { type: "stream_end" });
      stream.status = "completed";
      return;
    }

    // Check if this is a permanent error (404/403) that shouldn't be retried
    const isPermanentError =
      err instanceof Error &&
      (err as Error & { isPermanent?: boolean }).isPermanent;

    if (!isPermanentError && retryCount < MAX_RETRIES) {
      const retryDelay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount);
      console.log(
        `${logPrefix} Retrying in ${retryDelay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`,
      );
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
      return executeStreamInternal({
        ...options,
        retryCount: retryCount + 1,
      });
    }

    // Log permanent errors differently for debugging
    if (isPermanentError) {
      console.log(
        `${logPrefix} Stream failed permanently (task not found or access denied): ${(err as Error).message}`,
      );
    }

    stream.status = "error";
    stream.error = err instanceof Error ? err : new Error("Stream failed");
    notifySubscribers(stream, {
      type: "error",
      message: stream.error.message,
    });
  }
}

/**
 * Execute a new chat stream.
 *
 * Posts a message to create a new stream and reads SSE responses.
 */
export async function executeStream(
  stream: ActiveStream,
  message: string,
  isUserMessage: boolean,
  context?: { url: string; content: string },
  retryCount: number = 0,
): Promise<void> {
  return executeStreamInternal({
    stream,
    mode: "new",
    message,
    isUserMessage,
    context,
    retryCount,
  });
}

/**
 * Reconnect to an existing task stream.
 *
 * This is used when a client wants to resume receiving updates from a
 * long-running background task. Messages are replayed from the last_message_id
 * position, allowing clients to catch up on missed events.
 *
 * @param stream - The active stream state
 * @param taskId - The task ID to reconnect to
 * @param lastMessageId - The last message ID received (for replay)
 * @param retryCount - Current retry count
 */
export async function executeTaskReconnect(
  stream: ActiveStream,
  taskId: string,
  lastMessageId: string = INITIAL_MESSAGE_ID,
  retryCount: number = 0,
): Promise<void> {
  return executeStreamInternal({
    stream,
    mode: "reconnect",
    taskId,
    lastMessageId,
    retryCount,
  });
}
