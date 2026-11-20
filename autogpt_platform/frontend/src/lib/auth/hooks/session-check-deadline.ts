import * as Sentry from "@sentry/nextjs";

type SessionCheck = "getCurrentUser" | "validateSession" | "refreshSession";

// A session check that never answers must still end, or the store's
// one-check slot stays taken and the tab never checks its session again
// (#16988). The deadline is well above prod's observed p99, and a missed one
// counts as unavailable: the user stays signed in and the store retries.
//
// It frees the store, not Next's action queue, which runs one server action
// at a time. A dropped action never ran, so the next check goes straight out.
// Checks still on Next's action queue can wait behind an unanswered action
// until a navigation discards it (see #17028).
export const SESSION_CHECK_DEADLINE_MS = 10_000;

// The name of the deadline's timer function, `sessionCheckDeadline` below.
// The auth interleavings harness tells deadlines from the store's other
// timers by it.
export const SESSION_CHECK_DEADLINE_TIMER = "sessionCheckDeadline";

class SessionCheckTimeoutError extends Error {
  constructor(check: SessionCheck) {
    super(`${check} got no answer in time`);
    this.name = "SessionCheckTimeoutError";
  }
}

export function isSessionCheckTimeout(error: unknown): boolean {
  return error instanceof SessionCheckTimeoutError;
}

export function withDeadline<T>(
  check: SessionCheck,
  answer: Promise<T>,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    function settle() {
      settled = true;
      clearTimeout(timer);
    }
    function sessionCheckDeadline() {
      reportTimeout(check);
      reject(new SessionCheckTimeoutError(check));
    }
    answer.then(
      (value) => {
        settle();
        resolve(value);
      },
      (error: unknown) => {
        settle();
        reject(error);
      },
    );
    // Armed once the current turn ends: every page that mounts `useAuth` runs a
    // check, and code stepping timers in the same turn (fake-timer tests that
    // render the app) must not meet a deadline for an answer already on its way.
    queueMicrotask(() => {
      if (!settled) {
        timer = setTimeout(sessionCheckDeadline, SESSION_CHECK_DEADLINE_MS);
      }
    });
  });
}

// Sentry captures console warnings, so timeouts skip the console and are
// reported here: a breadcrumb each time, and a warning for the first of each
// kind per page load, since a tab stuck behind an unanswered action times out
// again on every retry and refocus.
const reportedTimeouts = new Set<SessionCheck>();

function reportTimeout(check: SessionCheck) {
  Sentry.addBreadcrumb({
    category: "session-check",
    level: "warning",
    message: `${check} got no answer in time`,
  });
  if (reportedTimeouts.has(check)) return;
  reportedTimeouts.add(check);
  Sentry.captureMessage("Session check got no answer in time", {
    level: "warning",
    tags: { context: "session-check", server_action: check },
  });
}
