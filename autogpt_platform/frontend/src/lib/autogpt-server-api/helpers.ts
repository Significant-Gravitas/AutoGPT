import {
  API_KEY_HEADER_NAME,
  IMPERSONATION_HEADER_NAME,
} from "@/lib/constants";
import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";

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

  // Filter out undefined values to prevent them from being included as "undefined" strings
  const filteredQuery = Object.entries(query).reduce(
    (acc, [key, value]) => {
      if (value !== undefined) {
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

export function createRequestHeaders(
  token: string,
  hasRequestBody: boolean,
  contentType: string = "application/json",
  originalRequest?: Request,
): Record<string, string> {
  const headers: Record<string, string> = {};

  if (hasRequestBody) {
    headers["Content-Type"] = contentType;
  }

  if (token && token !== "no-token-found") {
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

// NOTE: makeAuthenticatedRequest and makeAuthenticatedFileUpload have been
// moved to server-auth.ts to avoid bundling server-only code with client code.
// Import them from "@/lib/autogpt-server-api/server-auth" when needed.
