import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;

  try {
    const token = await getServerAuthToken();
    const backendUrl = environment.getAGPTServerBaseUrl();
    const cancelUrl = new URL(
      `/api/chat/sessions/${sessionId}/cancel`,
      backendUrl,
    );

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (token && token !== "no-token-found") {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(cancelUrl.toString(), {
      method: "POST",
      headers,
    });

    const text = await response.text();
    try {
      const data = JSON.parse(text);
      return new Response(JSON.stringify(data), {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    } catch {
      return new Response(
        JSON.stringify({
          cancelled: false,
          reason: "invalid_backend_response",
        }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      );
    }
  } catch (error) {
    console.error("Cancel proxy error:", error);
    return new Response(
      JSON.stringify({
        cancelled: false,
        reason: "proxy_error",
        detail: error instanceof Error ? error.message : String(error),
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
}
