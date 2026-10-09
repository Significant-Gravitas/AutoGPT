interface RequestFailure {
  status: number;
  method: string;
  url: string;
  errorMessage: string;
  responseData: unknown;
}

// A failing endpoint is usually polled or retried, so the same failure lands in
// the console many times a minute on top of the browser's own network line.
const DEDUPE_WINDOW_MS = 10_000;

/**
 * Log one client-side request failure, collapsing repeats of the same
 * (method, url, status) into a single line plus a count.
 */
export function logClientRequestFailure(failure: RequestFailure) {
  const key = `${failure.method} ${failure.url} ${failure.status}`;
  const open = openWindows.get(key);

  if (open) {
    open.suppressed += 1;
    return;
  }

  emit(failure);
  openWindows.set(key, {
    suppressed: 0,
    timer: setTimeout(() => {
      const closed = openWindows.get(key);
      openWindows.delete(key);
      if (closed?.suppressed) {
        logAtSeverityOf(failure.status)(
          `Request failed on client ×${closed.suppressed + 1} in the last ${DEDUPE_WINDOW_MS / 1000}s`,
          { status: failure.status, method: failure.method, url: failure.url },
        );
      }
    }, DEDUPE_WINDOW_MS),
  });
}

// The open windows outlive a test file; reset them between cases.
export function resetClientRequestFailureLog() {
  openWindows.forEach(({ timer }) => clearTimeout(timer));
  openWindows.clear();
}

interface OpenWindow {
  suppressed: number;
  timer: ReturnType<typeof setTimeout>;
}

const openWindows = new Map<string, OpenWindow>();

// Statuses the app asks for and handles: an unauthenticated read of a gated
// resource, a permission the user does not have, an optional resource that is
// absent. A 4xx that means the request itself was wrong (400, 422) and every
// 5xx stay errors.
const EXPECTED_STATUSES = new Set([401, 403, 404]);

function emit({
  status,
  method,
  url,
  errorMessage,
  responseData,
}: RequestFailure) {
  logAtSeverityOf(status)("Request failed on client", {
    status,
    method,
    url,
    errorMessage,
    responseData: responseData || "No response data",
  });
}

function logAtSeverityOf(status: number) {
  return EXPECTED_STATUSES.has(status) ? console.warn : console.error;
}
