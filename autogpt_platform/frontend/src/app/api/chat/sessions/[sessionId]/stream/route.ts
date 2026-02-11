import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
} as const;

/**
 * Normalize backend SSE events so they conform to the AI SDK's strict schema.
 *
 * The AI SDK uses `z.strictObject` for its stream event union, which rejects
 * unknown keys. The backend's `StreamError` may include extra fields (`code`,
 * `details`) that cause Zod validation failures. This transform strips those
 * extra fields from `error` events before forwarding.
 */
function normalizeSSEStream(
  input: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  return input.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        // Keep the last part in the buffer (it may be incomplete)
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const normalized = normalizeSSEEvent(part);
          controller.enqueue(encoder.encode(normalized + "\n\n"));
        }
      },
      flush(controller) {
        if (buffer.trim()) {
          const normalized = normalizeSSEEvent(buffer);
          controller.enqueue(encoder.encode(normalized + "\n\n"));
        }
      },
    }),
  );
}

/**
 * Normalize a single SSE event string. For `error` type events, strip
 * fields that the AI SDK schema does not expect (e.g. `code`, `details`).
 */
function normalizeSSEEvent(event: string): string {
  const lines = event.split("\n");
  const dataLines: string[] = [];
  const otherLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      dataLines.push(line.slice(6));
    } else {
      otherLines.push(line);
    }
  }

  if (dataLines.length === 0) return event;

  const dataStr = dataLines.join("");
  try {
    const parsed = JSON.parse(dataStr) as Record<string, unknown>;
    if (parsed.type === "error") {
      // Only keep the fields the AI SDK schema expects
      const normalized = {
        type: "error",
        errorText:
          typeof parsed.errorText === "string"
            ? parsed.errorText
            : "An unexpected error occurred",
      };
      const newData = `data: ${JSON.stringify(normalized)}`;
      return [...otherLines.filter((l) => l.length > 0), newData].join("\n");
    }
  } catch {
    // Not valid JSON — pass through as-is
  }

  return event;
}

/**
 * SSE Proxy for chat streaming.
 * Supports POST with context (page content + URL) in the request body.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;

  try {
    const body = await request.json();
    const { message, is_user_message, context } = body;

    if (!message) {
      return new Response(
        JSON.stringify({ error: "Missing message parameter" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    // Get auth token from server-side session
    const token = await getServerAuthToken();

    // Build backend URL
    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(
      `/api/chat/sessions/${sessionId}/stream`,
      backendUrl,
    );

    // Forward request to backend with auth header
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    };

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(streamUrl.toString(), {
      method: "POST",
      headers,
      body: JSON.stringify({
        message,
        is_user_message: is_user_message ?? true,
        context: context || null,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return new Response(error, {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!response.body) {
      return new Response(
        JSON.stringify({ error: "Empty response from chat service" }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(normalizeSSEStream(response.body), {
      headers: SSE_HEADERS,
    });
  } catch (error) {
    console.error("SSE proxy error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to connect to chat service",
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

/**
 * Resume an active stream for a session.
 *
 * Called by the AI SDK's `useChat(resume: true)` on page load.
 * Proxies to the backend which checks for an active stream and either
 * replays it (200 + SSE) or returns 204 No Content.
 */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;

  try {
    const token = await getServerAuthToken();

    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(
      `/api/chat/sessions/${sessionId}/stream`,
      backendUrl,
    );

    const headers: Record<string, string> = {
      Accept: "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    };

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(streamUrl.toString(), {
      method: "GET",
      headers,
    });

    // 204 = no active stream to resume
    if (response.status === 204) {
      return new Response(null, { status: 204 });
    }

    if (!response.ok) {
      const error = await response.text();
      return new Response(error, {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!response.body) {
      return new Response(null, { status: 204 });
    }

    return new Response(normalizeSSEStream(response.body), {
      headers: {
        ...SSE_HEADERS,
        "x-vercel-ai-ui-message-stream": "v1",
      },
    });
  } catch (error) {
    console.error("Resume stream proxy error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to connect to chat service",
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
