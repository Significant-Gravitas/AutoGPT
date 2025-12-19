/**
 * Server-only authentication helpers.
 *
 * This file can ONLY be imported from server-side code (API routes,
 * Server Components, Server Actions). It uses next/headers.
 *
 * For client-side code, use the proxy route at /api/proxy/* which
 * handles authentication server-side.
 */

import "server-only";

import { getServerAuthToken as getAuthTokenFromAuth } from "@/lib/auth/server/getServerAuth";
import {
  ApiError,
  buildUrlWithQuery,
  createRequestHeaders,
  isAuthenticationError,
  isLogoutInProgress,
  parseApiError,
  parseApiResponse,
  serializeRequestBody,
} from "./helpers";

// Re-export buildServerUrl for convenience
export { buildServerUrl, buildUrlWithQuery } from "./helpers";

/**
 * Get the server-side authentication token.
 * Returns a placeholder string if no token is available.
 */
export async function getServerAuthToken(): Promise<string> {
  try {
    const token = await getAuthTokenFromAuth();
    if (!token) {
      return "no-token-found";
    }
    return token;
  } catch (error) {
    console.error("Failed to get auth token:", error);
    return "no-token-found";
  }
}

/**
 * Get authentication headers for server-side requests.
 */
export async function getServerAuthHeaders(
  hasRequestBody: boolean,
  contentType: string = "application/json",
): Promise<Record<string, string>> {
  const token = await getServerAuthToken();
  return createRequestHeaders(token, hasRequestBody, contentType);
}

/**
 * Make an authenticated request to the backend.
 * Server-side only - uses next/headers for authentication.
 */
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

/**
 * Upload a file with authentication.
 * Server-side only - uses next/headers for authentication.
 */
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
