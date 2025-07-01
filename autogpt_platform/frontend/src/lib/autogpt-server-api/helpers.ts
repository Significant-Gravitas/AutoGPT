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
  console.log("🔐 Getting server auth token...");

  const supabase = await getServerSupabase();

  if (!supabase) {
    console.error("❌ Supabase client not available");
    throw new Error("Supabase client not available");
  }

  console.log("✅ Supabase client obtained");

  try {
    const {
      data: { session },
      error,
    } = await supabase.auth.getSession();

    console.log("🔍 Session check:", {
      hasSession: !!session,
      hasAccessToken: !!session?.access_token,
      hasError: !!error,
      errorMessage: error?.message,
      userId: session?.user?.id,
    });

    if (error || !session?.access_token) {
      console.warn("⚠️ No valid session or access token found");
      return "no-token-found";
    }

    console.log("✅ Valid access token obtained");
    return session.access_token;
  } catch (error) {
    console.error("❌ Failed to get auth token:", error);
    return "no-token-found";
  }
}

export function createRequestHeaders(
  token: string,
  hasRequestBody: boolean,
  contentType: string = "application/json",
): Record<string, string> {
  console.log("📋 Creating request headers:", {
    hasToken: token !== "no-token-found",
    tokenPreview:
      token !== "no-token-found"
        ? `${token.substring(0, 10)}...`
        : "no-token-found",
    hasRequestBody,
    contentType,
  });

  const headers: Record<string, string> = {};

  if (hasRequestBody) {
    headers["Content-Type"] = contentType;
  }

  if (token && token !== "no-token-found") {
    headers["Authorization"] = `Bearer ${token}`;
  }

  console.log("🏷️ Headers created:", {
    headerKeys: Object.keys(headers),
    hasAuth: !!headers["Authorization"],
    hasContentType: !!headers["Content-Type"],
  });

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
    const logoutTimestamp = window.localStorage.getItem("supabase-logout");
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
  console.log("🔐 === AUTHENTICATED REQUEST START ===");
  console.log("📋 Request details:", {
    method,
    url,
    contentType,
    hasPayload: !!payload,
    payloadPreview: payload
      ? JSON.stringify(payload).substring(0, 100) + "..."
      : "No payload",
  });

  const token = await getServerAuthToken();
  const payloadAsQuery = ["GET", "DELETE"].includes(method);
  const hasRequestBody = !payloadAsQuery && payload !== undefined;

  console.log("🎯 Request configuration:", {
    payloadAsQuery,
    hasRequestBody,
    tokenAvailable: token !== "no-token-found",
  });

  const headers = createRequestHeaders(token, hasRequestBody, contentType);
  const body = hasRequestBody
    ? serializeRequestBody(payload, contentType)
    : undefined;

  console.log("📡 Making fetch request to:", url);
  console.log(
    "📤 Request body preview:",
    body ? String(body).substring(0, 200) + "..." : "No body",
  );

  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
    });

    console.log("📨 Response received:", {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      headers: Object.fromEntries(response.headers.entries()),
      url: response.url,
    });

    if (!response.ok) {
      const errorDetail = await parseApiError(response);
      console.error("❌ Request failed:", {
        status: response.status,
        statusText: response.statusText,
        errorDetail,
      });

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

      // For other errors, throw as normal
      throw new Error(errorDetail);
    }

    const responseData = await parseApiResponse(response);
    console.log("✅ Request completed successfully");
    console.log(
      "📦 Response data preview:",
      responseData
        ? JSON.stringify(responseData).substring(0, 200) + "..."
        : "No data",
    );
    console.log("🏁 === AUTHENTICATED REQUEST END ===");

    return responseData;
  } catch (error) {
    console.error("💥 === AUTHENTICATED REQUEST FAILED ===");
    console.error("❌ Fetch error:", error);
    console.error("❌ Error details:", {
      name: error instanceof Error ? error.name : "Unknown",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : "No stack trace",
    });
    throw error;
  }
}

export async function makeAuthenticatedFileUpload(
  url: string,
  formData: FormData,
): Promise<string> {
  console.log("📁 === AUTHENTICATED FILE UPLOAD START ===");
  console.log("📋 Upload details:", {
    url,
    formDataEntries: Array.from(formData.entries()).map(([key, value]) => ({
      key,
      value:
        value instanceof File
          ? `File: ${value.name} (${value.size} bytes)`
          : value,
    })),
  });

  const token = await getServerAuthToken();

  const headers: Record<string, string> = {};
  if (token && token !== "no-token-found") {
    headers["Authorization"] = `Bearer ${token}`;
  }

  console.log("🏷️ Upload headers:", {
    hasAuth: !!headers["Authorization"],
    headerKeys: Object.keys(headers),
  });

  console.log("📡 Making file upload request to:", url);

  try {
    // Don't set Content-Type for FormData - let the browser set it with boundary
    const response = await fetch(url, {
      method: "POST",
      headers,
      body: formData,
    });

    console.log("📨 Upload response received:", {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      headers: Object.fromEntries(response.headers.entries()),
    });

    if (!response.ok) {
      // Handle authentication errors gracefully for file uploads too
      const errorMessage = `Error uploading file: ${response.statusText}`;
      console.error("❌ File upload failed:", {
        status: response.status,
        statusText: response.statusText,
        errorMessage,
      });

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

      throw new Error(errorMessage);
    }

    const responseText = await response.text();
    console.log("✅ File upload completed successfully");
    console.log(
      "📦 Upload response preview:",
      responseText.substring(0, 200) + "...",
    );
    console.log("🏁 === AUTHENTICATED FILE UPLOAD END ===");

    return responseText;
  } catch (error) {
    console.error("💥 === AUTHENTICATED FILE UPLOAD FAILED ===");
    console.error("❌ Upload error:", error);
    console.error("❌ Error details:", {
      name: error instanceof Error ? error.name : "Unknown",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : "No stack trace",
    });
    throw error;
  }
}
