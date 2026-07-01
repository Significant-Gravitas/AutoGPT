import {
  API_KEY_HEADER_NAME,
  IMPERSONATION_HEADER_NAME,
} from "@/lib/constants";
import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";
import { cache } from "react";

import { GraphValidationErrorResponse } from "./types";

export class ApiError<R = any> extends Error {
  public status: number;
  public response: R;

  constructor(message: string, status: number, response: R) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.response = response;
  }

  /**
   * Type guard to check if this error is a structured graph validation error
   */
  isGraphValidationError(): this is ApiError<GraphValidationErrorResponse> {
    return (
      this.response !== undefined &&
      typeof this.response === "object" &&
      this.response !== null &&
      "detail" in this.response &&
      typeof this.response.detail === "object" &&
      this.response.detail !== null &&
      "type" in this.response.detail &&
      this.response.detail.type === "validation_error" &&
      "node_errors" in this.response.detail &&
      typeof this.response.detail.node_errors === "object"
    );
  }
}

export function buildRequestUrl(
  baseUrl: string,
  path: string,
  method: string,
  payload?: Record<string, any>,
): string {
  const url = baseUrl + path;
  const payloadAsQuery = ["GET", "DELETE"].includes(method);

  if (payloadAsQuery && payload) {
    return buildUrlWithQuery(url, payload);
  }

  return url;
}

export function buildClientUrl(path: string): string {
  return `/api/proxy/api${path}`;
}

export function buildServerUrl(path: string): string {
  return `${environment.getAGPTServerApiUrl()}${path}`;
}

export function buildUrlWithQuery(
  url: string,
  query?: Record<string, any>,
): string {
  if (!query) return url;

  // Drop null/undefined so URLSearchParams doesn't serialize them as the
  // strings "null" / "undefined".
  const filteredQuery = Object.entries(query).reduce(
    (acc, [key, value]) => {
      if (value != null) {
        acc[key] = value;
      }
      return acc;
    },
    {} as Record<string, any>,
  );

  const queryParams = new URLSearchParams(filteredQuery);
  return queryParams.size > 0 ? `${url}?${queryParams.toString()}` : url;
}

export async function handleFetchError(response: Response): Promise<ApiError> {
  const errorMessage = await parseApiError(response);

  // Safely parse response body - it might not be JSON (e.g., HTML error pages)
  let responseData: any = null;
  try {
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      responseData = await response.json();
    } else {
      // For non-JSON responses, get the text content
      responseData = await response.text();
    }
  } catch (e) {
    // If parsing fails, use null as response data
    console.warn("Failed to parse error response body:", e);
    responseData = null;
  }

  return new ApiError(
    errorMessage || "Request failed",
    response.status,
    responseData,
  );
}

// JWT-per-session cache so every proxied backend call doesn't re-request a
// token. Entries expire 5 minutes before the JWT itself does.
const TOKEN_EXPIRY_MARGIN_MS = 5 * 60 * 1000;
const MAX_TOKEN_CACHE_ENTRIES = 1000;
const serverTokenCache = new Map<
  string,
  { token: string; expiresAt: number }
>();

export function readJwtExpiryMs(token: string): number {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split(".")[1], "base64url").toString("utf-8"),
    );
    if (typeof payload.exp === "number") return payload.exp * 1000;
  } catch {
    // fall through to a conservative default
  }
  return Date.now() + TOKEN_EXPIRY_MARGIN_MS * 2;
}

export function cacheServerToken(sessionCookie: string, token: string): void {
  if (serverTokenCache.size >= MAX_TOKEN_CACHE_ENTRIES) {
    const oldestKey = serverTokenCache.keys().next().value;
    if (oldestKey) serverTokenCache.delete(oldestKey);
  }
  serverTokenCache.set(sessionCookie, {
    token,
    expiresAt: readJwtExpiryMs(token) - TOKEN_EXPIRY_MARGIN_MS,
  });
}

export function getCachedServerToken(sessionCookie: string): string | null {
  const cached = serverTokenCache.get(sessionCookie);
  if (cached && cached.expiresAt > Date.now()) return cached.token;
  return null;
}

/**
 * Mints (or returns a cached) backend-API JWT for the current request's
 * session by calling the Better Auth token endpoint on this same server.
 * The Python backend validates the JWT against /api/auth/jwks.
 *
 * Deliberately uses an HTTP call instead of importing the Better Auth server
 * instance: this module is part of the client component graph (via the orval
 * mutator), where transitively importing pg/nodemailer breaks the browser
 * bundle. Cookies are read via next/headers `cookies()` (lazily required, as
 * the previous Supabase client did here) so that a session cookie set
 * earlier in the SAME server action — e.g. right after sign-in — is visible
 * immediately, not just on the next request.
 */
export const getServerAuthToken = cache(async (): Promise<string | null> => {
  if (environment.isClientSide()) {
    // Browser requests go through /api/proxy, which attaches the token
    // server-side; there is no client-side token.
    return null;
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const headersModule = require("next/headers");
    const nextHeaders = headersModule as typeof import("next/headers");
    const cookieStore = await nextHeaders.cookies();

    const sessionCookie = cookieStore
      .getAll()
      .find(
        ({ name }) =>
          name === "better-auth.session_token" ||
          name === "__Secure-better-auth.session_token",
      );
    if (!sessionCookie) return null;

    const cached = getCachedServerToken(sessionCookie.value);
    if (cached) return cached;

    const cookieHeader = cookieStore
      .getAll()
      .map(({ name, value }) => `${name}=${encodeURIComponent(value)}`)
      .join("; ");

    const baseURL =
      process.env.BETTER_AUTH_URL ||
      process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
      "http://localhost:3000";
    const response = await fetch(`${baseURL}/api/auth/token`, {
      headers: { cookie: cookieHeader },
      cache: "no-store",
    });
    if (!response.ok) return null;

    const { token } = (await response.json()) as { token?: string };
    if (!token) return null;

    cacheServerToken(sessionCookie.value, token);
    return token;
  } catch (error) {
    console.error("Failed to get auth token:", error);
    return null;
  }
});

export function createRequestHeaders(
  token: string | null,
  hasRequestBody: boolean,
  contentType: string = "application/json",
  originalRequest?: Request,
): Record<string, string> {
  const headers: Record<string, string> = {};

  if (hasRequestBody) {
    headers["Content-Type"] = contentType;
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Forward admin impersonation header if present
  if (originalRequest) {
    const impersonationHeader = originalRequest.headers.get(
      IMPERSONATION_HEADER_NAME,
    );
    if (impersonationHeader) {
      headers[IMPERSONATION_HEADER_NAME] = impersonationHeader;
    }

    // Forward X-API-Key header if present
    const apiKeyHeader = originalRequest.headers.get(API_KEY_HEADER_NAME);
    if (apiKeyHeader) {
      headers[API_KEY_HEADER_NAME] = apiKeyHeader;
    }

    // Forward Sentry distributed-tracing headers so the backend transaction
    // continues the browser span instead of starting a disconnected trace.
    for (const name of ["sentry-trace", "baggage"] as const) {
      const value = originalRequest.headers.get(name);
      if (value) {
        headers[name] = value;
      }
    }
  }

  return headers;
}

export function serializeRequestBody(
  payload: any,
  contentType: string = "application/json",
): string {
  switch (contentType) {
    case "application/json":
      return JSON.stringify(payload);
    case "application/x-www-form-urlencoded":
      return new URLSearchParams(payload).toString();
    default:
      // For custom content types, assume payload is already properly formatted
      return typeof payload === "string" ? payload : JSON.stringify(payload);
  }
}

export async function parseApiError(response: Response): Promise<string> {
  // Handle 413 Payload Too Large with user-friendly message
  if (response.status === 413) {
    return "File is too large — max size is 256MB";
  }

  try {
    const errorData = await response.clone().json();

    if (
      Array.isArray(errorData.detail) &&
      errorData.detail.length > 0 &&
      errorData.detail[0].loc
    ) {
      // Pydantic validation error
      const errors = errorData.detail.map((err: any) => {
        const location = err.loc.join(" -> ");
        return `${location}: ${err.msg}`;
      });
      return errors.join("\n");
    }

    if (typeof errorData.detail === "object" && errorData.detail !== null) {
      if (errorData.detail.message) return errorData.detail.message;
      return response.statusText; // Fallback to status text if no message
    }

    // Check for file size error from backend
    if (
      typeof errorData.detail === "string" &&
      errorData.detail.includes("exceeds the maximum")
    ) {
      const match = errorData.detail.match(/maximum allowed size of (\d+)MB/);
      const maxSize = match ? match[1] : "256";
      return `File is too large — max size is ${maxSize}MB`;
    }

    return errorData.detail || errorData.error || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function parseApiResponse(response: Response): Promise<any> {
  // Handle responses with no content
  if (
    response.status === 204 ||
    response.headers.get("Content-Length") === "0"
  ) {
    return null;
  }

  try {
    return await response.json();
  } catch (e) {
    if (e instanceof SyntaxError) {
      return null;
    }
    throw e;
  }
}

export function isAuthenticationError(
  response: Response,
  errorDetail: string,
): boolean {
  return (
    response.status === 401 ||
    response.status === 403 ||
    errorDetail.toLowerCase().includes("not authenticated") ||
    errorDetail.toLowerCase().includes("unauthorized") ||
    errorDetail.toLowerCase().includes("authentication failed")
  );
}

export function isLogoutInProgress(): boolean {
  if (environment.isServerSide()) return false;

  try {
    // Check if logout was recently triggered
    const logoutTimestamp = storage.get(Key.LOGOUT);
    if (logoutTimestamp) {
      const timeDiff = Date.now() - parseInt(logoutTimestamp);
      // Consider logout in progress for 5 seconds after trigger
      return timeDiff < 5000;
    }

    // Check if we're being redirected to login
    return (
      window.location.pathname.includes("/login") ||
      window.location.pathname.includes("/logout")
    );
  } catch {
    return false;
  }
}

export async function makeAuthenticatedRequest(
  method: string,
  url: string,
  payload?: Record<string, any>,
  contentType: string = "application/json",
  originalRequest?: Request,
): Promise<any> {
  const token = await getServerAuthToken();
  const payloadAsQuery = ["GET", "DELETE"].includes(method);
  const hasRequestBody = !payloadAsQuery && payload !== undefined;

  // Add query parameters for GET/DELETE requests
  let requestUrl = url;
  if (payloadAsQuery && payload) {
    requestUrl = buildUrlWithQuery(url, payload);
  }

  const response = await fetch(requestUrl, {
    method,
    headers: createRequestHeaders(
      token,
      hasRequestBody,
      contentType,
      originalRequest,
    ),
    body: hasRequestBody
      ? serializeRequestBody(payload, contentType)
      : undefined,
  });

  if (!response.ok) {
    const errorDetail = await parseApiError(response);

    // Try to parse the full response body for better error context
    let responseData = null;
    try {
      responseData = await response.clone().json();
    } catch {
      // Ignore parsing errors
    }

    // Handle authentication errors gracefully during logout
    if (isAuthenticationError(response, errorDetail)) {
      if (isLogoutInProgress()) {
        // Silently return null during logout to prevent error noise
        console.debug(
          "Authentication request failed during logout, ignoring:",
          errorDetail,
        );
      }
    }

    // For other errors, throw ApiError with proper status code
    throw new ApiError(errorDetail, response.status, responseData);
  }

  return parseApiResponse(response);
}

export async function makeAuthenticatedFileUpload(
  url: string,
  formData: FormData,
  originalRequest?: Request,
): Promise<string> {
  const token = await getServerAuthToken();

  // Reuse existing header creation logic but exclude Content-Type for FormData
  const headers = createRequestHeaders(
    token,
    false,
    "application/json",
    originalRequest,
  );

  // Don't set Content-Type for FormData - let the browser set it with boundary
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    // Handle authentication errors gracefully for file uploads too
    const errorMessage = `Error uploading file: ${response.statusText}`;

    // Try to parse error response
    let responseData = null;
    try {
      const responseText = await response.clone().text();
      if (responseText) {
        responseData = JSON.parse(responseText);
      }
    } catch {
      // Ignore parsing errors
    }

    if (response.status === 401 || response.status === 403) {
      if (isLogoutInProgress()) {
        console.debug(
          "File upload authentication failed during logout, ignoring",
        );
        return "";
      }
      console.warn("File upload authentication failed:", errorMessage);
      return "";
    }

    throw new ApiError(errorMessage, response.status, responseData);
  }

  return await response.json();
}
