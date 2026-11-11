import {
  ApiError,
  createRequestHeaders,
  getServerAuthToken,
} from "@/lib/autogpt-server-api/helpers";
import * as Sentry from "@sentry/nextjs";

import { handleRefusedAccount } from "@/lib/auth/account-banned";
import { getSystemHeaders } from "@/lib/impersonation";
import { getDatafastAttribution } from "@/services/analytics/datafast-attribution";
import { getPostHogSessionHeader } from "@/services/analytics/posthog-request-context";
import { environment } from "@/services/environment";
import { parseJsonExactWithIntegers } from "@/lib/json-text";
import { transformDates } from "./date-transformer";
import { recordIntegers, restoreIntegers } from "./integer-provenance";
import {
  fetchWithFailureLog,
  logClientRequestFailure,
  logServerRequestFailure,
} from "./request-failure-log";

function getBaseURL(): string {
  if (!environment.isServerSide()) {
    return "/api/proxy";
  } else {
    return environment.getAGPTServerBaseUrl();
  }
}

// Routes that return block inputs the app shows or sends back: graphs,
// executions ("Run again"), reviews (also on the home page), library agents
// and presets, schedules, store listing graphs and builder search results.
// A rounded integer from them would be saved or run even if never touched.
const EXACT_INTEGER_ROUTES = [
  "/api/graphs",
  "/api/executions",
  "/api/library/",
  "/api/review/",
  "/api/store/listings/versions/",
  "/api/schedules",
  "/api/builder/search",
  "/api/home",
];

function keepsBigIntegers(url: string): boolean {
  return EXACT_INTEGER_ROUTES.some((route) => url.startsWith(route));
}

function parseExact(url: string) {
  return (text: string): unknown => {
    const { value, integers } = parseJsonExactWithIntegers(text);
    // Per browser tab: on the server the notes would be shared between users.
    if (environment.isClientSide()) recordIntegers(url, value, integers);
    return value;
  };
}

const getBody = async <T>(
  c: Response | Request,
  parseJson?: (text: string) => unknown,
): Promise<T> => {
  // 204 No Content responses (and 200s with Content-Length: 0) have no body.
  // Calling .json() on them throws "Unexpected end of JSON input" because the
  // backend may still set Content-Type: application/json on 204s. Short-circuit
  // to null so callers see a normal success rather than a parse error.
  if (
    "status" in c &&
    (c.status === 204 || c.headers.get("Content-Length") === "0")
  ) {
    return null as T;
  }

  const contentType = c.headers.get("content-type");

  if (contentType && contentType.includes("application/json")) {
    return parseJson ? (parseJson(await c.text()) as T) : c.json();
  }

  if (
    contentType &&
    (contentType.includes("application/pdf") ||
      contentType.includes("application/zip") ||
      contentType.startsWith("image/") ||
      contentType.startsWith("video/"))
  ) {
    return c.blob() as Promise<T>;
  }

  return c.text() as Promise<T>;
};

export const customMutator = async <
  T extends { data: any; status: number; headers: Headers },
>(
  url: string,
  options: RequestInit,
): Promise<T> => {
  const requestOptions = options;
  const method = (requestOptions.method || "GET") as
    | "GET"
    | "POST"
    | "PUT"
    | "DELETE"
    | "PATCH";
  const data =
    typeof requestOptions.body === "string" && environment.isClientSide()
      ? restoreIntegers(method, url, requestOptions.body)
      : requestOptions.body;
  let headers: Record<string, string> = {
    ...((requestOptions.headers as Record<string, string>) || {}),
  };

  if (environment.isClientSide()) {
    const traceData = Sentry.getTraceData?.() ?? {};
    for (const [key, value] of Object.entries(traceData)) {
      if (typeof value === "string") {
        headers[key] = value;
      }
    }
    Object.assign(headers, getSystemHeaders());
    Object.assign(headers, getDatafastAttribution());
    Object.assign(headers, getPostHogSessionHeader());
  }

  const isFormData = data instanceof FormData;
  const contentType = isFormData ? "multipart/form-data" : "application/json";

  // Currently, only two content types are handled here: application/json and multipart/form-data
  // For POST/PUT/PATCH requests, always set Content-Type to application/json if not FormData
  // This is required by the proxy even for requests without a body
  if (
    !isFormData &&
    !headers["Content-Type"] &&
    ["POST", "PUT", "PATCH"].includes(method)
  ) {
    headers["Content-Type"] = "application/json";
  }

  const baseUrl = getBaseURL();

  // The caching in React Query in our system depends on the url, so the base_url could be different for the server and client sides.
  // here url also contains encoded query params
  const fullUrl = `${baseUrl}${url}`;

  if (environment.isServerSide()) {
    try {
      const token = await getServerAuthToken();
      const authHeaders = createRequestHeaders(token, !!data, contentType);
      headers = { ...headers, ...authHeaders };
    } catch (error) {
      console.warn("Failed to get server auth token:", error);
    }
  }

  const response = await fetchWithFailureLog(fullUrl, {
    ...requestOptions,
    method,
    headers,
    body: data,
  });

  // Check if response is a redirect (3xx) and redirect is allowed
  const allowRedirect = requestOptions.redirect !== "error";
  const isRedirect = response.status >= 300 && response.status < 400;

  // For redirect responses, return early without trying to parse body
  if (allowRedirect && isRedirect) {
    return {
      status: response.status,
      data: null,
      headers: response.headers,
    } as T;
  }

  if (!response.ok) {
    let responseData: any = null;
    try {
      responseData = await getBody<any>(response);
    } catch (error) {
      console.warn("Failed to parse error response body:", error);
      responseData = { error: "Failed to parse response" };
    }

    const rawDetail = responseData?.detail;
    const detail = Array.isArray(rawDetail)
      ? rawDetail
          .map((e: unknown) =>
            e && typeof e === "object" && "msg" in e
              ? String((e as { msg: unknown }).msg)
              : JSON.stringify(e),
          )
          .join("; ")
      : typeof rawDetail === "string"
        ? rawDetail
        : typeof rawDetail?.message === "string"
          ? rawDetail.message
          : rawDetail != null
            ? JSON.stringify(rawDetail)
            : undefined;

    const errorMessage =
      detail ||
      responseData?.message ||
      response.statusText ||
      `HTTP ${response.status}`;

    const failure = {
      status: response.status,
      method,
      url: fullUrl.replace(baseUrl, ""), // Show relative URL for cleaner logs
      errorMessage,
      responseData,
    };

    if (environment.isServerSide()) {
      logServerRequestFailure(failure);
    } else {
      logClientRequestFailure(failure);
      handleRefusedAccount(response.status, responseData);
    }

    throw new ApiError(errorMessage, response.status, responseData);
  }

  const responseData = await getBody<unknown>(
    response,
    keepsBigIntegers(url) ? parseExact(url) : undefined,
  );

  // Transform ISO date strings to Date objects in the response data
  const transformedData =
    responseData instanceof Blob ? responseData : transformDates(responseData);

  return {
    status: response.status,
    data: transformedData,
    headers: response.headers,
  } as T;
};
