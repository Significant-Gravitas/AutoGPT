import type { Event } from "@sentry/nextjs";

// Next.js logs exactly this when a client-side navigation cannot fetch the RSC
// payload (flaky network, tab backgrounded, deploy in progress). It then does a
// full browser navigation to the same URL, so the user sees nothing wrong.
// See next/dist/client/components/router-reducer/fetch-server-response.
const NEXT_RSC_FALLBACK_MESSAGE =
  /^Failed to fetch RSC payload for .+\. Falling back to browser navigation\.$/;

/**
 * True for the Sentry event that captureConsoleIntegration builds from Next's
 * handled RSC fetch fallback. The event's own title is the underlying
 * "TypeError: Failed to fetch", so `ignoreErrors` cannot tell it apart from a
 * real fetch failure in our code; the console text in `extra.arguments` can.
 */
export function isNextRSCNavigationFallback(event: Event) {
  if (event.logger !== "console") return false;
  const args = event.extra?.arguments;
  if (!Array.isArray(args)) return false;
  const [message] = args;
  return typeof message === "string" && NEXT_RSC_FALLBACK_MESSAGE.test(message);
}

const CREDENTIAL_HEADERS = new Set(["cookie", "authorization"]);

// Keeps the request's body, cookies and credential headers out of server and
// edge events.
export function stripRequestCredentials<T extends Event>(event: T): T {
  const { request } = event;
  if (!request) return event;
  delete request.data;
  delete request.cookies;
  if (!request.headers) return event;
  request.headers = Object.fromEntries(
    Object.entries(request.headers).filter(
      ([name]) => !CREDENTIAL_HEADERS.has(name.toLowerCase()),
    ),
  );
  return event;
}
