import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

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

    // Return the SSE stream directly
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
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

    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
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
