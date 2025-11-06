import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

/**
 * SSE Proxy for chat streaming.
 * EventSource doesn't support custom headers, so we need a server-side proxy
 * that adds authentication and forwards the SSE stream to the client.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;
  const searchParams = request.nextUrl.searchParams;
  const message = searchParams.get("message");
  const isUserMessage = searchParams.get("is_user_message");

  if (!message) {
    return new Response("Missing message parameter", { status: 400 });
  }

  try {
    // Get auth token from server-side session
    const token = await getServerAuthToken();

    // Build backend URL
    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(
      `/api/chat/sessions/${sessionId}/stream`,
      backendUrl,
    );
    streamUrl.searchParams.set("message", message);

    // Pass is_user_message parameter if provided
    if (isUserMessage !== null) {
      streamUrl.searchParams.set("is_user_message", isUserMessage);
    }

    // Forward request to backend with auth header
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
