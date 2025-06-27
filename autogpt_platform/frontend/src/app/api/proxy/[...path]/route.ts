import { NextRequest, NextResponse } from "next/server";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_BASE_URL || "http://localhost:8006";

/**
 * A simple proxy route that forwards requests to the backend API.
 * It injects the server-side authentication token into the Authorization header.
 * It streams the request and response bodies to be efficient with memory.
 */
async function handler(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  // Construct the backend URL from the incoming request.
  const { path } = await params;
  const url = new URL(req.url);
  const queryString = url.search;
  const backendPath = path.join("/");
  const backendUrl = `${BACKEND_BASE_URL}/${backendPath}${queryString}`;

  const requestHeaders = new Headers(req.headers);

  const token = await getServerAuthToken();
  if (token) {
    requestHeaders.set("Authorization", `Bearer ${token}`);
  }

  // The 'host' header is automatically set by fetch and should not be copied.
  requestHeaders.delete("host");

  try {
    // Forward the request to the backend, streaming the body.
    const backendResponse = await fetch(backendUrl, {
      method: req.method,
      headers: requestHeaders,
      body: req.body,
    });

    // Return the response from the backend directly to the client.
    return backendResponse;
  } catch (error) {
    console.error("API proxy error:", error);
    const detail =
      error instanceof Error ? error.message : "An unknown error occurred";
    return NextResponse.json(
      { error: "Proxy request failed", detail },
      { status: 502 }, // Bad Gateway
    );
  }
}

export {
  handler as GET,
  handler as POST,
  handler as PUT,
  handler as PATCH,
  handler as DELETE,
};
