import { fetch } from "expo/fetch";

import { parseJSONLoose } from "../lib/json";
import {
  AbortedError,
  HttpError,
  NetworkError,
  RedirectRefusedError,
} from "./errors";

export type HttpResponse = Awaited<ReturnType<typeof fetch>>;

export interface HttpRequest {
  url: string;
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  headers?: Record<string, string>;
  json?: unknown;
  signal?: AbortSignal;
  /** Bounds the wait for response headers; streaming bodies are not limited. */
  timeoutMs?: number;
}

export interface OpenResponse {
  response: HttpResponse;
  /** Stops forwarding the caller's abort once the body has been consumed. */
  release: () => void;
}

const DEFAULT_TIMEOUT_MS = 20_000;

/**
 * Every request is sent without the shared cookie jar and without following
 * redirects. A bearer credential may only reach the exact origin it was
 * issued for, and a stored cookie must never stand in for it.
 */
export async function openRequest(request: HttpRequest): Promise<OpenResponse> {
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort();
  const external = request.signal;
  if (external?.aborted) throw new AbortedError();
  external?.addEventListener("abort", forwardAbort);
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, request.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const release = () => external?.removeEventListener("abort", forwardAbort);

  const headers: Record<string, string> = {
    Accept: "application/json",
    ...request.headers,
  };
  let body: string | undefined;
  if (request.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(request.json);
  }

  try {
    const response = await fetch(request.url, {
      method: request.method ?? "GET",
      headers,
      body,
      signal: controller.signal,
      credentials: "omit",
      redirect: "manual",
    });
    if (response.status >= 300 && response.status < 400) {
      release();
      throw new RedirectRefusedError();
    }
    return { response, release };
  } catch (error) {
    release();
    if (error instanceof RedirectRefusedError) throw error;
    if (external?.aborted) throw new AbortedError();
    throw new NetworkError(timedOut);
  } finally {
    clearTimeout(timer);
  }
}

/** Reads a whole body as JSON when possible, otherwise as text. */
export async function readBody(
  response: HttpResponse,
  signal?: AbortSignal,
): Promise<unknown> {
  try {
    const text = await response.text();
    return text ? parseJSONLoose(text) : null;
  } catch {
    if (signal?.aborted) throw new AbortedError();
    throw new NetworkError(false);
  }
}

export function retryAfterSeconds(response: HttpResponse): number | null {
  const raw =
    response.headers.get("retry-after") ??
    response.headers.get("x-retry-after");
  if (!raw) return null;
  const seconds = Number(raw);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

export async function httpErrorFrom(
  response: HttpResponse,
  signal?: AbortSignal,
) {
  const detail = await readBody(response, signal).catch(() => null);
  return new HttpError({
    status: response.status,
    detail,
    retryAfterSeconds: retryAfterSeconds(response),
  });
}

/** One JSON round trip; throws HttpError for non-success statuses. */
export async function requestJSON<T>(
  request: HttpRequest,
): Promise<{ data: T; response: HttpResponse }> {
  const { response, release } = await openRequest(request);
  try {
    if (!response.ok) throw await httpErrorFrom(response, request.signal);
    if (response.status === 204) return { data: null as T, response };
    return { data: (await readBody(response, request.signal)) as T, response };
  } finally {
    release();
  }
}
