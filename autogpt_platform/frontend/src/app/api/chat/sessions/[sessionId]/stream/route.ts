import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";
import { normalizeSSEStream, SSE_HEADERS } from "../../../sse-helpers";

export const maxDuration = 800;

const DEBUG_SSE_TIMEOUT_MS = process.env.NEXT_PUBLIC_SSE_TIMEOUT_MS
  ? Number(process.env.NEXT_PUBLIC_SSE_TIMEOUT_MS)
  : undefined;

function debugSignal(): AbortSignal | undefined {
  if (!DEBUG_SSE_TIMEOUT_MS) return undefined;
  console.warn(
    `[SSE_DEBUG] Simulating proxy timeout in ${DEBUG_SSE_TIMEOUT_MS}ms`,
  );
  return AbortSignal.timeout(DEBUG_SSE_TIMEOUT_MS);
}

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

    const token = await getServerAuthToken();

    const backendUrl = environment.getAGPTServerBaseUrl();
    const streamUrl = new URL(
      `/api/chat/sessions/${sessionId}/stream`,
      backendUrl,
    );

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
      signal: debugSignal(),
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
      signal: debugSignal(),
    });

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
