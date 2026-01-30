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

export async function executeStream(
  stream: ActiveStream,
  message: string,
  isUserMessage: boolean,
  context?: { url: string; content: string },
  retryCount: number = 0,
): Promise<void> {
  const { sessionId, abortController } = stream;

  try {
    const url = `/api/chat/sessions/${sessionId}/stream`;
    const body = JSON.stringify({
      message,
      is_user_message: isUserMessage,
      context: context || null,
    });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body,
      signal: abortController.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
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

            notifySubscribers(stream, chunk);

            if (chunk.type === "stream_end") {
              stream.status = "completed";
              return;
            }

            if (chunk.type === "error") {
              stream.status = "error";
              stream.error = new Error(
                chunk.message || chunk.content || "Stream error",
              );
              return;
            }
          } catch (err) {
            console.warn("[StreamExecutor] Failed to parse SSE chunk:", err);
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

    if (retryCount < MAX_RETRIES) {
      const retryDelay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount);
      console.log(
        `[StreamExecutor] Retrying in ${retryDelay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`,
      );
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
      return executeStream(
        stream,
        message,
        isUserMessage,
        context,
        retryCount + 1,
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
  lastMessageId: string = "0",
  retryCount: number = 0,
): Promise<void> {
  const { abortController } = stream;

  try {
    const url = `/api/chat/tasks/${taskId}/stream?last_message_id=${encodeURIComponent(lastMessageId)}`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "text/event-stream",
      },
      signal: abortController.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
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

            notifySubscribers(stream, chunk);

            if (chunk.type === "stream_end") {
              stream.status = "completed";
              return;
            }

            if (chunk.type === "error") {
              stream.status = "error";
              stream.error = new Error(
                chunk.message || chunk.content || "Stream error",
              );
              return;
            }
          } catch (err) {
            console.warn(
              "[StreamExecutor] Failed to parse task reconnect SSE chunk:",
              err,
            );
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

    if (retryCount < MAX_RETRIES) {
      const retryDelay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount);
      console.log(
        `[StreamExecutor] Task reconnect retrying in ${retryDelay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`,
      );
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
      return executeTaskReconnect(stream, taskId, lastMessageId, retryCount + 1);
    }

    stream.status = "error";
    stream.error = err instanceof Error ? err : new Error("Task reconnect failed");
    notifySubscribers(stream, {
      type: "error",
      message: stream.error.message,
    });
  }
}
