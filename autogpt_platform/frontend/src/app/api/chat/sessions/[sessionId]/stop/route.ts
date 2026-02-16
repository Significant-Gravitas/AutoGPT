import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

/**
 * Proxy for stopping an active chat generation.
 * Forwards to the backend POST /api/chat/sessions/{sessionId}/stop endpoint.
 */
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;

  try {
    const token = await getServerAuthToken();

    const backendUrl = environment.getAGPTServerBaseUrl();
    const stopUrl = new URL(
      `/api/chat/sessions/${sessionId}/stop`,
      backendUrl,
    );

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(stopUrl.toString(), {
      method: "POST",
      headers,
    });

    const data = await response.text();

    return new Response(data, {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Stop proxy error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to stop chat generation",
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
