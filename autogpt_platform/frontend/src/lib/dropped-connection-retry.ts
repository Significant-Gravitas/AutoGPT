import { routeTemplate } from "@/lib/route-template";

// Node's fetch (undici) reuses keep-alive connections, and one can be dead by
// the time it's picked up: something on the way forgot it, or the far side
// closed it as the request went out. A serverless instance is frozen between
// requests, so undici's idle timers don't run and a thawed instance can reuse
// a connection that's long gone. undici fails the request with "fetch failed",
// drops the dead connection and doesn't retry. Like nginx, send a read once
// more, and never a request that might change something.
const RETRYABLE_METHODS: ReadonlySet<string> = new Set(["GET", "HEAD"]);

// What undici gives as the cause when the connection closed or was reset under
// a request. Timeouts are not here: the caller's deadline covers both tries.
const DROPPED_CONNECTION_CODES: ReadonlySet<string> = new Set([
  "ECONNRESET",
  "EPIPE",
  "UND_ERR_SOCKET",
]);

export async function fetchRetryingDroppedConnection(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (error) {
    const method = (init.method ?? "GET").toUpperCase();
    const code = getFetchFailureCode(error);
    if (!RETRYABLE_METHODS.has(method) || !isDroppedConnection(error, code)) {
      throw error;
    }
    // During a server render, Next's fetch answers an identical GET that has
    // no signal with the first call's promise, which here is the failure. A
    // signal opts out of that, and the caller's own keeps its cancellation.
    const response = await fetch(url, {
      ...init,
      signal: init.signal ?? new AbortController().signal,
    });
    // Templated: a path can carry a capability token, such as a share link's.
    const path = routeTemplate(new URL(url, "http://localhost").pathname);
    console.info(
      `Retried ${method} ${path} after a dropped connection (${code}): ${response.status}`,
    );
    return response;
  }
}

/** The code of a failure below HTTP, such as "ECONNRESET", if there is one. */
export function getFetchFailureCode(error: unknown): string | undefined {
  if (!(error instanceof Error)) return undefined;
  const { cause } = error;
  if (typeof cause !== "object" || cause === null || !("code" in cause)) {
    return undefined;
  }
  return typeof cause.code === "string" ? cause.code : undefined;
}

// "fetch failed" means no response arrived at all; a body cut off after the
// headers fails with "terminated" instead, and is never sent again.
function isDroppedConnection(error: unknown, code: string | undefined) {
  return (
    error instanceof TypeError &&
    error.message === "fetch failed" &&
    code !== undefined &&
    DROPPED_CONNECTION_CODES.has(code)
  );
}
