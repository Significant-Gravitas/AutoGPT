import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { environment } from "@/services/environment";
import { NextRequest } from "next/server";

/**
 * Stop an active chat stream for a session.
 *
 * Proxies to backend with server-side auth token injection.
 */
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;

  try {
    const token = await getServerAuthToken();

    const backendUrl = environment.getAGPTServerBaseUrl();
    const stopUrl = new URL(`/api/chat/sessions/${sessionId}/stop`, backendUrl);

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

    const bodyText = await response.text();

    return new Response(bodyText || null, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: "Failed to stop chat stream",
        detail: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
