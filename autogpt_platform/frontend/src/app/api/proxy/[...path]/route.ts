import {
  ApiError,
  getServerAuthToken,
  makeAuthenticatedFileUpload,
  makeAuthenticatedRequest,
} from "@/lib/autogpt-server-api/helpers";
import { environment } from "@/services/environment";
import { NextRequest, NextResponse } from "next/server";

// Increase body size limit to 256MB to match backend file upload limit
export const maxDuration = 300; // 5 minutes timeout for large uploads
export const dynamic = "force-dynamic";

import {
  fetchWorkspaceDownloadWithRetry,
  getWorkspaceDownloadErrorMessage,
  isWorkspaceDownloadRequest,
} from "./route.helpers";

function buildBackendUrl(path: string[], queryString: string): string {
  const backendPath = path.join("/");
  return `${environment.getAGPTServerBaseUrl()}/${backendPath}${queryString}`;
}

/**
 * Handle workspace file download requests with proper binary response streaming.
 */
async function handleWorkspaceDownload(
  req: NextRequest,
  backendUrl: string,
): Promise<NextResponse> {
  const token = await getServerAuthToken();

  const headers: Record<string, string> = {};
  if (token && token !== "no-token-found") {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetchWorkspaceDownloadWithRetry(
    backendUrl,
    headers,
    2,
    500,
  );

  if (!response.ok) {
    return await createWorkspaceDownloadErrorResponse(response);
  }

  // Fully buffer the response before forwarding.  Passing response.body as a
  // ReadableStream causes silent truncation in Next.js / Vercel — the last
  // ~10 KB of larger files are dropped, corrupting PNGs and truncating CSVs.
  const buffer = await response.arrayBuffer();

  const contentType =
    response.headers.get("Content-Type") || "application/octet-stream";
  const contentDisposition = response.headers.get("Content-Disposition");

  const responseHeaders: Record<string, string> = {
    "Content-Type": contentType,
    "Content-Length": String(buffer.byteLength),
  };

  if (contentDisposition) {
    responseHeaders["Content-Disposition"] = contentDisposition;
  }

  return new NextResponse(buffer, {
    status: 200,
    headers: responseHeaders,
  });
}

async function createWorkspaceDownloadErrorResponse(
  response: Response,
): Promise<NextResponse> {
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? "";

  try {
    if (contentType.includes("application/json")) {
      const body = await response.json();
      return NextResponse.json(body, { status: response.status });
    }

    const text = await response.text();
    const detail =
      getWorkspaceDownloadErrorMessage(text) ||
      response.statusText ||
      "Failed to download file";

    return NextResponse.json({ detail }, { status: response.status });
  } catch {
    return NextResponse.json(
      {
        detail: response.statusText || "Failed to download file",
      },
      { status: response.status },
    );
  }
}

async function handleJsonRequest(
  req: NextRequest,
  method: string,
  backendUrl: string,
): Promise<any> {
  let payload;

  try {
    payload = await req.json();
  } catch (error) {
    // Handle cases where request body is empty, invalid JSON, or already consumed
    console.warn("Failed to parse JSON from request body:", error);
    payload = null;
  }

  return await makeAuthenticatedRequest(
    method,
    backendUrl,
    payload,
    "application/json",
    req,
  );
}

async function handleFormDataRequest(
  req: NextRequest,
  backendUrl: string,
): Promise<any> {
  const formData = await req.formData();
  return await makeAuthenticatedFileUpload(backendUrl, formData, req);
}

async function handleUrlEncodedRequest(
  req: NextRequest,
  method: string,
  backendUrl: string,
): Promise<any> {
  const textPayload = await req.text();
  const params = new URLSearchParams(textPayload);
  const payload = Object.fromEntries(params.entries());
  return await makeAuthenticatedRequest(
    method,
    backendUrl,
    payload,
    "application/x-www-form-urlencoded",
    req,
  );
}

async function handleGetDeleteRequest(
  method: string,
  backendUrl: string,
  req: NextRequest,
): Promise<any> {
  return await makeAuthenticatedRequest(
    method,
    backendUrl,
    undefined,
    "application/json",
    req,
  );
}

function createUnsupportedContentTypeResponse(
  contentType: string | null,
): NextResponse {
  return NextResponse.json(
    {
      error:
        "Unsupported Content-Type for proxying with authentication helpers.",
      receivedContentType: contentType,
      supportedContentTypes: [
        "application/json",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
      ],
    },
    { status: 415 },
  );
}

function createResponse(
  responseBody: any,
  responseStatus: number,
  responseHeaders: Record<string, string>,
): NextResponse {
  if (responseStatus === 204) {
    return new NextResponse(null, { status: responseStatus });
  } else {
    return NextResponse.json(responseBody, {
      status: responseStatus,
      headers: responseHeaders,
    });
  }
}

function createErrorResponse(
  error: unknown,
  path: string,
  method: string,
): NextResponse {
  if (
    error &&
    typeof error === "object" &&
    "status" in error &&
    [401, 403].includes(error.status as number)
  ) {
    // Log this since it indicates a potential frontend bug
    console.warn(
      `Authentication error in API proxy for ${method} ${path}:`,
      "message" in error ? error.message : error,
    );
  }

  // If it's our custom ApiError, preserve the original status and response
  if (error instanceof ApiError) {
    return NextResponse.json(error.response || { error: error.message }, {
      status: error.status,
    });
  }

  // For JSON parsing errors, provide more context
  if (error instanceof SyntaxError && error.message.includes("JSON")) {
    return NextResponse.json(
      {
        error: "Invalid response from backend",
        detail: error.message ?? "Backend returned non-JSON response",
      },
      { status: 502 },
    );
  }

  // For other errors, use generic response
  const detail =
    error instanceof Error ? error.message : "An unknown error occurred";
  return NextResponse.json(
    { error: "Proxy request failed", detail },
    { status: 500 },
  );
}

/**
 * A simple proxy route that forwards requests to the backend API.
 * It injects the server-side authentication token into the Authorization header.
 * It uses the makeAuthenticatedRequest and makeAuthenticatedFileUpload helpers
 * to handle request body parsing and authentication.
 */
async function handler(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const url = new URL(req.url);
  const queryString = url.search;
  const backendUrl = buildBackendUrl(path, queryString);

  const method = req.method;
  const contentType = req.headers.get("Content-Type");

  let responseBody: any;
  const responseHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };

  try {
    // Handle workspace file downloads separately (binary response)
    if (method === "GET" && isWorkspaceDownloadRequest(path)) {
      return await handleWorkspaceDownload(req, backendUrl);
    }

    if (method === "GET" || method === "DELETE") {
      responseBody = await handleGetDeleteRequest(method, backendUrl, req);
    } else if (contentType?.includes("application/json")) {
      responseBody = await handleJsonRequest(req, method, backendUrl);
    } else if (contentType?.includes("multipart/form-data")) {
      responseBody = await handleFormDataRequest(req, backendUrl);
    } else if (contentType?.includes("application/x-www-form-urlencoded")) {
      responseBody = await handleUrlEncodedRequest(req, method, backendUrl);
    } else {
      return createUnsupportedContentTypeResponse(contentType);
    }

    return createResponse(responseBody, 200, responseHeaders);
  } catch (error) {
    return createErrorResponse(
      error,
      path.map((s) => `/${s}`).join(""),
      method,
    );
  }
}

export {
  handler as DELETE,
  handler as GET,
  handler as PATCH,
  handler as POST,
  handler as PUT,
};
