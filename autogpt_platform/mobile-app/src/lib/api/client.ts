import { API_URL } from "@/lib/env";
import { clearAccessToken, getAccessToken } from "@/lib/auth/token";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
}

export async function authHeaders(): Promise<Record<string, string>> {
  return { Authorization: `Bearer ${await getAccessToken()}` };
}

export function apiUrl(path: string, query?: RequestOptions["query"]) {
  const params = Object.entries(query ?? {})
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${encodeURIComponent(String(value))}`);
  return `${API_URL}${path}${params.length ? `?${params.join("&")}` : ""}`;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}) {
  let response = await send(path, options);
  if (response.status === 401) {
    clearAccessToken();
    response = await send(path, options);
  }
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function send(path: string, { method = "GET", body, query }: RequestOptions) {
  return fetch(apiUrl(path, query), {
    method,
    headers: {
      ...(await authHeaders()),
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

async function readErrorMessage(response: Response) {
  try {
    const data = (await response.json()) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
  } catch {
    // Not JSON: fall through to the status line.
  }
  return `Request failed (${response.status})`;
}
