import {
  ApiError,
  createRequestHeaders,
  getServerAuthToken,
} from "@/lib/autogpt-server-api/helpers";

import { transformDates } from "./date-transformer";
import { environment } from "@/services/environment";

const FRONTEND_BASE_URL =
  process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";
const API_PROXY_BASE_URL = `${FRONTEND_BASE_URL}/api/proxy`; // Sending request via nextjs Server

const getBaseUrl = (): string => {
  if (!environment.isServerSide()) {
    return API_PROXY_BASE_URL;
  } else {
    return environment.getAGPTServerBaseUrl();
  }
};

const getBody = <T>(c: Response | Request): Promise<T> => {
  const contentType = c.headers.get("content-type");

  if (contentType && contentType.includes("application/json")) {
    return c.json();
  }

  if (contentType && contentType.includes("application/pdf")) {
    return c.blob() as Promise<T>;
  }

  return c.text() as Promise<T>;
};

export const customMutator = async <
  T extends { data: any; status: number; headers: Headers },
>(
  url: string,
  options: RequestInit & {
    params?: any;
  } = {},
): Promise<T> => {
  const { params, ...requestOptions } = options;
  const method = (requestOptions.method || "GET") as
    | "GET"
    | "POST"
    | "PUT"
    | "DELETE"
    | "PATCH";
  const data = requestOptions.body;
  let headers: Record<string, string> = {
    ...((requestOptions.headers as Record<string, string>) || {}),
  };

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

  const queryString = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  const baseUrl = getBaseUrl();

  // The caching in React Query in our system depends on the url, so the base_url could be different for the server and client sides.
  const fullUrl = `${baseUrl}${url}${queryString}`;

  if (environment.isServerSide()) {
    try {
      const token = await getServerAuthToken();
      const authHeaders = createRequestHeaders(token, !!data, contentType);
      headers = { ...headers, ...authHeaders };
    } catch (error) {
      console.warn("Failed to get server auth token:", error);
    }
  }

  const response = await fetch(fullUrl, {
    ...requestOptions,
    method,
    headers,
    body: data,
  });

  if (!response.ok) {
    const response_data = await getBody<any>(response);
    const errorMessage =
      response_data?.detail || response_data?.message || response.statusText;

    console.error(
      `Request failed ${environment.isServerSide() ? "on server" : "on client"}`,
      { status: response.status, url: fullUrl, data: response_data },
    );

    throw new ApiError(errorMessage, response.status, response_data);
  }

  const response_data = await getBody<T["data"]>(response);

  // Transform ISO date strings to Date objects in the response data
  const transformedData = transformDates(response_data);

  return {
    status: response.status,
    data: transformedData,
    headers: response.headers,
  } as T;
};
