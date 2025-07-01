import { NextRequest, NextResponse } from "next/server";
import {
  makeAuthenticatedRequest,
  makeAuthenticatedFileUpload,
} from "@/lib/autogpt-server-api/helpers";

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_BASE_URL || "http://localhost:8006";

console.log("🔧 Proxy Route - Backend Base URL:", BACKEND_BASE_URL);

function buildBackendUrl(path: string[], queryString: string): string {
  const backendPath = path.join("/");
  const finalUrl = `${BACKEND_BASE_URL}/${backendPath}${queryString}`;
  console.log("🔗 Building backend URL:", {
    path,
    backendPath,
    queryString,
    finalUrl,
  });
  return finalUrl;
}

async function handleJsonRequest(
  req: NextRequest,
  method: string,
  backendUrl: string,
): Promise<any> {
  console.log("📝 Handling JSON request:", { method, backendUrl });
  const payload = await req.json();
  console.log("📄 Request payload:", payload);

  const result = await makeAuthenticatedRequest(
    method,
    backendUrl,
    payload,
    "application/json",
  );
  console.log("✅ JSON request completed successfully");
  return result;
}

async function handleFormDataRequest(
  req: NextRequest,
  backendUrl: string,
): Promise<any> {
  console.log("📁 Handling FormData request:", { backendUrl });
  const formData = await req.formData();
  console.log(
    "📄 FormData entries:",
    Array.from(formData.entries()).map(([key, value]) => ({
      key,
      value: value instanceof File ? `File: ${value.name}` : value,
    })),
  );

  const result = await makeAuthenticatedFileUpload(backendUrl, formData);
  console.log("✅ FormData request completed successfully");
  return result;
}

async function handleUrlEncodedRequest(
  req: NextRequest,
  method: string,
  backendUrl: string,
): Promise<any> {
  console.log("🔤 Handling URL encoded request:", { method, backendUrl });
  const textPayload = await req.text();
  const params = new URLSearchParams(textPayload);
  const payload = Object.fromEntries(params.entries());
  console.log("📄 URL encoded payload:", payload);

  const result = await makeAuthenticatedRequest(
    method,
    backendUrl,
    payload,
    "application/x-www-form-urlencoded",
  );
  console.log("✅ URL encoded request completed successfully");
  return result;
}

async function handleRequestWithoutBody(
  method: string,
  backendUrl: string,
): Promise<any> {
  console.log("🚀 Handling request without body:", { method, backendUrl });

  const result = await makeAuthenticatedRequest(method, backendUrl);
  console.log("✅ Request without body completed successfully");
  return result;
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
    { status: 415 }, // Unsupported Media Type
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

function createErrorResponse(error: unknown): NextResponse {
  console.error("❌ API proxy error:", error);
  console.error(
    "❌ Error stack:",
    error instanceof Error ? error.stack : "No stack trace",
  );
  console.error("❌ Error details:", {
    name: error instanceof Error ? error.name : "Unknown",
    message: error instanceof Error ? error.message : String(error),
    cause: error instanceof Error ? error.cause : undefined,
  });

  const detail =
    error instanceof Error ? error.message : "An unknown error occurred";
  return NextResponse.json(
    { error: "Proxy request failed", detail },
    { status: 500 }, // Internal Server Error
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
  console.log("🚀 === PROXY REQUEST START ===");

  const { path } = await params;
  const url = new URL(req.url);
  const queryString = url.search;

  console.log("📋 Request details:", {
    method: req.method,
    originalUrl: req.url,
    path,
    queryString,
    headers: Object.fromEntries(req.headers.entries()),
    nextUrl: url.toString(),
  });

  const backendUrl = buildBackendUrl(path, queryString);

  const method = req.method;
  const contentType = req.headers.get("Content-Type");

  console.log("🎯 Processing request:", {
    method,
    contentType,
    backendUrl,
  });

  let responseBody: any;
  const responseStatus: number = 200;
  const responseHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };

  try {
    if (method === "GET" || method === "DELETE") {
      console.log("🔄 Routing to handleRequestWithoutBody");
      responseBody = await handleRequestWithoutBody(method, backendUrl);
    } else if (contentType?.includes("application/json")) {
      console.log("🔄 Routing to handleJsonRequest");
      responseBody = await handleJsonRequest(req, method, backendUrl);
    } else if (contentType?.includes("multipart/form-data")) {
      console.log("🔄 Routing to handleFormDataRequest");
      responseBody = await handleFormDataRequest(req, backendUrl);
      responseHeaders["Content-Type"] = "text/plain";
    } else if (contentType?.includes("application/x-www-form-urlencoded")) {
      console.log("🔄 Routing to handleUrlEncodedRequest");
      responseBody = await handleUrlEncodedRequest(req, method, backendUrl);
    } else {
      console.log("❌ Unsupported content type:", contentType);
      return createUnsupportedContentTypeResponse(contentType);
    }

    console.log("✅ Request processed successfully:", {
      responseStatus,
      responseHeaders,
      responseBodyType: typeof responseBody,
      responseBodyPreview:
        typeof responseBody === "object"
          ? JSON.stringify(responseBody).substring(0, 200) + "..."
          : String(responseBody).substring(0, 200),
    });

    console.log("🏁 === PROXY REQUEST END ===");
    return createResponse(responseBody, responseStatus, responseHeaders);
  } catch (error) {
    console.log("💥 === PROXY REQUEST FAILED ===");
    return createErrorResponse(error);
  }
}

export {
  handler as GET,
  handler as POST,
  handler as PUT,
  handler as PATCH,
  handler as DELETE,
};
