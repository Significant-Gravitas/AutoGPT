import {
  API_KEY_HEADER_NAME,
  IMPERSONATION_HEADER_NAME,
} from "@/lib/constants";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { environment } from "@/services/environment";
import { NextRequest, NextResponse } from "next/server";

import {
  fetchWorkspaceDownloadWithRetry,
  getWorkspaceDownloadErrorMessage,
  isWorkspaceDownloadRequest,
} from "./route.helpers";

// 5 minutes for large uploads. Most calls finish in well under a second.
export const maxDuration = 300;
export const dynamic = "force-dynamic";

const FORWARDED_REQUEST_HEADERS: ReadonlyArray<string> = [
  "content-type",
  "content-length",
  "accept",
  "accept-language",
  IMPERSONATION_HEADER_NAME.toLowerCase(),
  API_KEY_HEADER_NAME.toLowerCase(),
  "sentry-trace",
  "baggage",
];

// Browsers advertise `zstd` in Accept-Encoding, but undici (Node 22's fetch)
// only auto-decompresses gzip/deflate/br. If we forwarded the browser value,
// Cloudflare in front of the backend would pick zstd, Node would hand us raw
// zstd bytes, the response handler would strip Content-Encoding, and Vercel's
// edge would then brotli-wrap those bytes — leaving the browser to decode br
// over zstd and fail JSON.parse on a 200 response.
const BACKEND_ACCEPT_ENCODING = "gzip, deflate, br";

// Headers that must not be forwarded from the backend to the client.
// Hop-by-hop entries are listed in RFC 7230 §6.1; content-encoding and
// content-length are stripped because Node's fetch auto-decompresses the
// backend body when reading response.body, so the streamed payload's actual
// byte length and encoding no longer match what the backend advertised.
// Letting Node recompute framing (via Transfer-Encoding: chunked) keeps
// HTTP parsing on the client side correct.
// `set-cookie` is stripped defensively so any backend cookie without an
// explicit Domain attribute can't attach to the frontend origin.
const STRIPPED_RESPONSE_HEADERS: ReadonlySet<string> = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "content-encoding",
  "content-length",
  "set-cookie",
]);

function buildBackendUrl(path: string[], queryString: string): string {
  return `${environment.getAGPTServerBaseUrl()}/${path.join("/")}${queryString}`;
}

function buildForwardHeaders(req: NextRequest, token: string | null): Headers {
  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = req.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("accept-encoding", BACKEND_ACCEPT_ENCODING);
  if (token) {
    headers.set("authorization", `Bearer ${token}`);
  }
  return headers;
}

function filterResponseHeaders(src: Headers): Headers {
  const out = new Headers();
  src.forEach((value, key) => {
    if (!STRIPPED_RESPONSE_HEADERS.has(key.toLowerCase())) {
      out.set(key, value);
    }
  });
  return out;
}

const METHODS_WITHOUT_BODY: ReadonlySet<string> = new Set([
  "GET",
  "HEAD",
  "DELETE",
  "OPTIONS",
]);

async function handleWorkspaceDownload(
  backendUrl: string,
  token: string | null,
): Promise<NextResponse> {
  const headers: Record<string, string> = {};
  if (token) {
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

  // Fully buffer the response before forwarding. Passing response.body as a
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

/**
 * Forwards browser API requests to the backend with the JWT pulled from
 * httpOnly cookies attached to the Authorization header.
 *
 * Body and response are streamed end-to-end so JSON / FormData / SSE bodies
 * pass through without parse → re-serialise round-trips. Status codes and
 * non-hop-by-hop response headers are preserved.
 *
 * Workspace file downloads have a dedicated buffering path because Vercel's
 * stream forwarding silently truncates large binary responses.
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

  try {
    const token = await getServerAuthToken();

    if (method === "GET" && isWorkspaceDownloadRequest(path)) {
      return await handleWorkspaceDownload(backendUrl, token);
    }

    const headers = buildForwardHeaders(req, token);
    const hasBody = !METHODS_WITHOUT_BODY.has(method);

    const backendResponse = await fetch(backendUrl, {
      method,
      headers,
      body: hasBody ? req.body : undefined,
      // `duplex: "half"` is required by Node's fetch when sending a
      // ReadableStream body. Cast because TS lib types haven't caught up.
      ...(hasBody ? ({ duplex: "half" } as RequestInit) : {}),
      redirect: "manual",
    });

    return new NextResponse(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: filterResponseHeaders(backendResponse.headers),
    });
  } catch (error) {
    console.error(`Proxy error for ${method} /${path.join("/")}:`, error);
    return NextResponse.json(
      {
        error: "Proxy request failed",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 },
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
