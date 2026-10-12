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

export const CLIENT_IGNORE_ERRORS = [
  // Sentry Replay (rrweb) serializing DOM snapshots with cross-origin
  // stylesheets (e.g. from browser extensions or CDN-loaded CSS)
  /Not allowed to access cross-origin stylesheet/,
  // Sentry SDK internal issue on some mobile browsers
  /Error invoking postEvent: Method not found/,
  // Cookiebot's own console warning when a crawler re-runs the page's scripts;
  // our layout includes it once (BUILDER-90X).
  /Cookiebot script is included twice/,
];

// Crawler user agents name themselves ("Googlebot/2.1", "GPTBot/1.0") and link
// to their docs ("+http://…"), which no real browser does. Headless and audit
// browsers identify themselves too. An empty user agent is never a person.
const BOT_USER_AGENT =
  /bot\/|bot;|\+https?:\/\/|crawler|spider|slurp|GoogleOther|Google-InspectionTool|HeadlessChrome|Lighthouse|PhantomJS|Prerender|python-requests|curl\/|wget\//i;

export function isFromBot(userAgent: string | undefined) {
  if (!userAgent?.trim()) return true;
  return BOT_USER_AGENT.test(userAgent);
}
