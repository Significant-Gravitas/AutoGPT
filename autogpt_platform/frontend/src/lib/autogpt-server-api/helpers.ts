import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";

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

    if (error || !session?.access_token) {
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

/**
 * Serialize request body based on content type
 */
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
    throw new Error(errorDetail);
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
    throw new Error(`Error uploading file: ${response.statusText}`);
  }

  // Parse the response
  return await response.text();
}
