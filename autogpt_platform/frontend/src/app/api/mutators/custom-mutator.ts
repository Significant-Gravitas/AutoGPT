import {
  createRequestHeaders,
  getServerAuthToken,
} from "@/lib/autogpt-server-api/helpers";
import { isServerSide } from "@/lib/utils/is-server-side";

const FRONTEND_BASE_URL =
  process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";
const API_PROXY_BASE_URL = `${FRONTEND_BASE_URL}/api/proxy`; // Sending request via nextjs Server

const getBaseUrl = (): string => {
  if (!isServerSide()) {
    return API_PROXY_BASE_URL;
  } else {
    return (
      process.env.NEXT_PUBLIC_AGPT_SERVER_BASE_URL || "http://localhost:8006"
    );
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

export const customMutator = async <T = any>(
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
  if (!isFormData && data && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const queryString = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  const baseUrl = getBaseUrl();

  // The caching in React Query in our system depends on the url, so the base_url could be different for the server and client sides.
  const fullUrl = `${baseUrl}${url}${queryString}`;

  if (isServerSide()) {
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

  // Error handling for server-side requests
  // We do not need robust error handling for server-side requests; we only need to log the error message and throw the error.
  // What happens if the server-side request fails?
  // 1. The error will be logged in the terminal, then.
  // 2. The error will be thrown, so the cached data for this particular queryKey will be empty, then.
  // 3. The client-side will send the request again via the proxy. If it fails again, the error will be handled on the client side.
  // 4. If the request succeeds on the server side, the data will be cached, and the client will use it instead of sending a request to the proxy.

  if (!response.ok && isServerSide()) {
    console.error("Request failed on server side", response);
    throw new Error(`Request failed with status ${response.status}`);
  }

  const response_data = await getBody<T>(response);

  return {
    status: response.status,
    data: response_data,
    headers: response.headers,
  } as T;
};
