import * as Sentry from "@sentry/nextjs";

interface RequestFailure {
  status: number;
  method: string;
  url: string;
  errorMessage: string;
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
        console.info(
          `${summarize(failure)} ×${closed.suppressed + 1} in the last ${DEDUPE_WINDOW_MS / 1000}s`,
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

// Outcomes the app asks for and handles: a signed-out read of a gated resource,
// a permission the user lacks (re-auth, email verification, waitlist), the
// paywall, an absent or deleted resource, and a request the user abandoned
// (the proxy answers 499 when the browser hangs up). They leave a breadcrumb
// for whatever error follows, but are not issues of their own.
const EXPECTED_STATUSES = new Set([401, 402, 403, 404, 499]);

// Long enough for a validation detail, short enough that an HTML error page
// from a proxy does not ride along.
const MAX_DETAIL_LENGTH = 300;

function emit(failure: RequestFailure) {
  const route = routeTemplate(failure.url);
  const message = summarize(failure);

  if (EXPECTED_STATUSES.has(failure.status)) {
    console.info(message);
    Sentry.addBreadcrumb({
      category: "http.client.failure",
      level: "info",
      message,
      data: { status: failure.status, method: failure.method, route },
    });
    return;
  }

  console.info(message, failure.errorMessage);
  Sentry.captureMessage(message, {
    level: failure.status >= 500 ? "error" : "warning",
    fingerprint: [
      "client-request-failure",
      `${failure.method} ${route}`,
      statusClass(failure.status),
    ],
    tags: {
      "http.method": failure.method,
      "http.route": route,
      "http.status_code": String(failure.status),
    },
    contexts: {
      request_failure: {
        method: failure.method,
        route,
        path: failure.url.split("?")[0],
        status: failure.status,
        detail: failure.errorMessage.slice(0, MAX_DETAIL_LENGTH),
      },
    },
  });
}

function summarize({ method, url, status }: RequestFailure) {
  return `Request failed on client: ${method} ${routeTemplate(url)} ${status}`;
}

function statusClass(status: number) {
  return status >= 100 && status < 600
    ? `${Math.floor(status / 100)}xx`
    : "0xx";
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const HEX_ID = /^[0-9a-f]{8,}$/i;

// An opaque ID (UUID, number, hash, nanoid) becomes {id} so every agent, run
// and session shares one route. A word or a short slug stays as it is.
function isOpaqueID(segment: string) {
  if (UUID.test(segment) || /^\d+$/.test(segment) || HEX_ID.test(segment)) {
    return true;
  }
  if (segment.includes("@") || segment.includes("%")) return true;
  return segment.length >= 12 && /\d/.test(segment);
}

export function routeTemplate(url: string) {
  const path = url.split(/[?#]/)[0] || "/";
  return path
    .split("/")
    .map((segment) => (segment && isOpaqueID(segment) ? "{id}" : segment))
    .join("/");
}
