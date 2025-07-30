import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { Key, storage } from "@/services/storage/local-storage";

export class ApiError extends Error {
  public status: number;
  public response?: any;

  constructor(message: string, status: number, response?: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.response = response;
  }
}

export function buildRequestUrl(
  baseUrl: string,
  path: string,
  method: string,
  payload?: Record<string, any>,
): string {
  let url = baseUrl + path;
  const payloadAsQuery = ["GET", "DELETE"].includes(method);

  if (payloadAsQuery && payload) {
    const queryParams = new URLSearchParams(payload);
    url += `?${queryParams.toString()}`;
  }

  return url;
}

export function buildClientUrl(path: string): string {
  return `/api/proxy/api${path}`;
}

export function buildServerUrl(path: string): string {
  const baseUrl =
    process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
  return `${baseUrl}${path}`;
}

export function buildUrlWithQuery(
  url: string,
  payload?: Record<string, any>,
): string {
  if (!payload) return url;

  const queryParams = new URLSearchParams(payload);
  return `${url}?${queryParams.toString()}`;
}

export function handleFetchError(response: Response, errorData: any): ApiError {
  return new ApiError(
    errorData?.error || "Request failed",
    response.status,
    errorData,
  );
}

export async function parseErrorResponse(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return { error: response.statusText };
  }
}

export async function getServerAuthToken(): Promise<string> {
  const supabase = await getServerSupabase();

  if (!supabase) {
    throw new Error("Supabase client not available");
  }

  try {
    const {
      data: { session },
      error,
    } = await supabase.auth.getSession();

    if (error || !session || !session.access_token) {
      return "no-token-found";
    }

    return session.access_token;
  } catch (error) {
    console.error("Failed to get auth token:", error);
    return "no-token-found";
  }
}

export function createRequestHeaders(
  token: string,
  hasRequestBody: boolean,
  contentType: string = "application/json",
): Record<string, string> {
  const headers: Record<string, string> = {};

  if (hasRequestBody) {
    headers["Content-Type"] = contentType;
  }

  if (token && token !== "no-token-found") {
    headers["Authorization"] = `Bearer ${token}`;
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
  try {
    const errorData = await response.json();

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

    return errorData.detail || response.statusText;
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

function isAuthenticationError(
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

function isLogoutInProgress(): boolean {
  if (typeof window === "undefined") return false;

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
): Promise<any> {
  const token = await getServerAuthToken();
  const payloadAsQuery = ["GET", "DELETE"].includes(method);
  const hasRequestBody = !payloadAsQuery && payload !== undefined;

  const response = await fetch(url, {
    method,
    headers: createRequestHeaders(token, hasRequestBody, contentType),
    body: hasRequestBody
      ? serializeRequestBody(payload, contentType)
      : undefined,
  });

  if (!response.ok) {
    const errorDetail = await parseApiError(response);

    // Try to parse the full response body for better error context
    let responseData = null;
    try {
      const responseText = await response.clone().text();
      if (responseText) {
        responseData = JSON.parse(responseText);
      }
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
        return null;
      }

      // For authentication errors outside logout, log but don't throw
      // This prevents crashes when session expires naturally
      console.warn("Authentication failed:", errorDetail);
      return null;
    }

    // For other errors, throw ApiError with proper status code
    throw new ApiError(errorDetail, response.status, responseData);
  }

  return parseApiResponse(response);
}

export async function makeAuthenticatedFileUpload(
  url: string,
  formData: FormData,
): Promise<string> {
  const token = await getServerAuthToken();

  const headers: Record<string, string> = {};
  if (token && token !== "no-token-found") {
    headers["Authorization"] = `Bearer ${token}`;
  }

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
