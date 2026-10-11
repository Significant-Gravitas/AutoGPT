import { ApiError } from "@/lib/autogpt-server-api/helpers";

// The message a failed `fetch` rejects with differs per engine: Chrome says
// "Failed to fetch", Safari "Load failed", Firefox "NetworkError when
// attempting to fetch resource.", and Node/undici "fetch failed".
const NETWORK_ERROR_MESSAGE =
  /failed to fetch|load failed|networkerror|network error|fetch failed|network request failed/i;

const TRANSIENT_HTTP_STATUSES = new Set([408, 425, 429]);

export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException || error instanceof Error) &&
    error.name === "AbortError"
  );
}

export function isNetworkError(error: unknown): boolean {
  return (
    error instanceof TypeError && NETWORK_ERROR_MESSAGE.test(error.message)
  );
}

/**
 * True when a request failed for a reason that says nothing about the request
 * itself: the network dropped, the fetch was cancelled (tab hidden, navigation
 * away), or the server was briefly unavailable. Retrying later is the right
 * response; telling the user something is broken is not.
 */
export function isTransientRequestError(error: unknown): boolean {
  if (isAbortError(error) || isNetworkError(error)) return true;
  if (error instanceof ApiError) {
    return error.status >= 500 || TRANSIENT_HTTP_STATUSES.has(error.status);
  }
  return false;
}

interface RetryOptions {
  attempts?: number;
  baseDelayMs?: number;
}

/**
 * Runs `fn`, retrying with exponential backoff while it fails with a
 * transient error. Aborts are rethrown at once (whoever cancelled the request
 * doesn't want it), as are other errors and the last transient one.
 */
export async function retryTransient<T>(
  fn: () => Promise<T>,
  { attempts = 3, baseDelayMs = 1_000 }: RetryOptions = {},
): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (
        attempt >= attempts ||
        isAbortError(error) ||
        !isTransientRequestError(error)
      ) {
        throw error;
      }
      await new Promise((resolve) =>
        setTimeout(resolve, baseDelayMs * 2 ** (attempt - 1)),
      );
    }
  }
}
