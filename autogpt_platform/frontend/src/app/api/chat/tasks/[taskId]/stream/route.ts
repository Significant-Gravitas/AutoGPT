import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

/**
 * SSE Proxy for task stream reconnection.
 *
 * This endpoint allows clients to reconnect to an ongoing or recently completed
 * background task's stream. It replays missed messages from Redis Streams and
 * subscribes to live updates if the task is still running.
 *
 * Client contract:
 * 1. When receiving an operation_started event, store the task_id
 * 2. To reconnect: GET /api/chat/tasks/{taskId}/stream?last_message_id={idx}
 * 3. Messages are replayed from the last_message_id position
 * 4. Stream ends when "finish" event is received
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const { taskId } = await params;
  const searchParams = request.nextUrl.searchParams;
  const lastMessageId = searchParams.get("last_message_id") || "0-0";

  try {
    // Get auth token from server-side session
    const token = await getServerAuthToken();

    // Build backend URL
    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(`/api/chat/tasks/${taskId}/stream`, backendUrl);
    streamUrl.searchParams.set("last_message_id", lastMessageId);

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
    console.error("Task stream proxy error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to connect to task stream",
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
