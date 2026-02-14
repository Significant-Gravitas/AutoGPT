import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";
import { normalizeSSEStream, SSE_HEADERS } from "../../../sse-helpers";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const { taskId } = await params;
  const searchParams = request.nextUrl.searchParams;
  const lastMessageId = searchParams.get("last_message_id") || "0-0";

  try {
    const token = await getServerAuthToken();

    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(`/api/chat/tasks/${taskId}/stream`, backendUrl);
    streamUrl.searchParams.set("last_message_id", lastMessageId);

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

    if (!response.body) {
      return new Response(null, { status: 204 });
    }

    return new Response(normalizeSSEStream(response.body), {
      headers: SSE_HEADERS,
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
