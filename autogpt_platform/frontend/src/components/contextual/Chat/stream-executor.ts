import { INITIAL_STREAM_ID } from "./chat-constants";
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

interface StreamExecutionOptions {
  stream: ActiveStream;
  mode: "new" | "reconnect";
  message?: string;
  isUserMessage?: boolean;
  context?: { url: string; content: string };
  taskId?: string;
  lastMessageId?: string;
  retryCount?: number;
}

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
    lastMessageId = INITIAL_STREAM_ID,
    retryCount = 0,
  } = options;

  const { sessionId, abortController } = stream;
  const isReconnect = mode === "reconnect";

  if (isReconnect) {
    if (!taskId) {
      throw new Error("taskId is required for reconnect mode");
    }
    if (lastMessageId === null || lastMessageId === undefined) {
      throw new Error("lastMessageId is required for reconnect mode");
    }
  } else {
    if (!message) {
      throw new Error("message is required for new stream mode");
    }
    if (isUserMessage === undefined) {
      throw new Error("isUserMessage is required for new stream mode");
    }
  }

  try {
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

    if (!response.ok) {
      const errorText = await response.text();
      let errorCode: string | undefined;
      let errorMessage = errorText || `HTTP ${response.status}`;
      try {
        const parsed = JSON.parse(errorText);
        if (parsed.detail) {
          const detail =
            typeof parsed.detail === "string"
              ? parsed.detail
              : parsed.detail.message || JSON.stringify(parsed.detail);
          errorMessage = detail;
          errorCode =
            typeof parsed.detail === "object" ? parsed.detail.code : undefined;
        }
      } catch {}

      const isPermanentError =
        isReconnect &&
        (response.status === 404 ||
          response.status === 403 ||
          response.status === 410);

      const error = new Error(errorMessage) as Error & {
        status?: number;
        isPermanent?: boolean;
        taskErrorCode?: string;
      };
      error.status = response.status;
      error.isPermanent = isPermanentError;
      error.taskErrorCode = errorCode;
      throw error;
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
          } catch {}
        }
      }
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      notifySubscribers(stream, { type: "stream_end" });
      stream.status = "completed";
      return;
    }

    const isPermanentError =
      err instanceof Error &&
      (err as Error & { isPermanent?: boolean }).isPermanent;

    if (!isPermanentError && retryCount < MAX_RETRIES) {
      const retryDelay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount);
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
      return executeStreamInternal({
        ...options,
        retryCount: retryCount + 1,
      });
    }

    stream.status = "error";
    stream.error = err instanceof Error ? err : new Error("Stream failed");
    notifySubscribers(stream, {
      type: "error",
      message: stream.error.message,
    });
  }
}

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

export async function executeTaskReconnect(
  stream: ActiveStream,
  taskId: string,
  lastMessageId: string = INITIAL_STREAM_ID,
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
