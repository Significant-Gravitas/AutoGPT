import { after } from "next/server";

/**
 * Better Auth's `advanced.backgroundTasks.handler`. Without one, Better Auth
 * awaits the emails it sends from sign-up, sign-in, password reset and email
 * change before it answers, so the response time gives away whether an email
 * went out: a sign-up for an address still waiting on its link waits on the
 * mail call, one for a verified address does not, though both get the same
 * body. With a handler the send runs alongside and the response does not wait.
 *
 * Better Auth already catches and logs a failed send at those call sites, so
 * nothing the response says depends on the email either way. The resend
 * endpoint does not come through here: it still awaits the send, so its
 * button can report a failure.
 *
 * `after` keeps the server alive until the send settles. Outside a Next
 * request scope (tests, scripts) it throws, and the send, already running,
 * finishes on its own.
 */
export function runAfterResponse(task: Promise<unknown>) {
  const settled = task.catch((error: unknown) => {
    console.error("Auth background task failed", {
      error: error instanceof Error ? error.message : String(error),
    });
  });
  try {
    after(settled);
  } catch {
    // No request scope: nothing to keep alive.
  }
}
