import type { Event, init } from "@sentry/nextjs";

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

// Script some browsers inject into every page (HeyTap Browser), and scrapers
// driving one (QQ Browser, Playwright), walk the page's iframes and log, in
// Chinese, "cannot access the iframe's content" for each cross-origin one the
// browser rightly refuses (BUILDER-943, 942, 8KH). None of it is our code.
const INJECTED_IFRAME_PROBE_MESSAGE = /^无法访问\s*iframe/i;

/**
 * True for that console warning. The event's title is the browser's own
 * "SecurityError: Blocked a frame … from accessing a cross-origin frame",
 * which our code could raise too, so only the injected script's console text
 * in `extra.arguments` marks it.
 */
export function isInjectedIframeProbe(event: Event) {
  if (event.logger !== "console") return false;
  if (event.exception?.values?.[0]?.type !== "SecurityError") return false;
  const args = event.extra?.arguments;
  if (!Array.isArray(args)) return false;
  const [message] = args;
  return (
    typeof message === "string" && INJECTED_IFRAME_PROBE_MESSAGE.test(message)
  );
}

export const CLIENT_IGNORE_ERRORS = [
  // Session replay (rrweb) serializing DOM snapshots with cross-origin
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

// Credentials, and the headers proxies put the visitor's IP address (and
// Vercel the location it resolves from it) in.
const PRIVATE_HEADERS = new Set([
  "cookie",
  "authorization",
  "x-forwarded-for",
  "x-real-ip",
  "x-vercel-forwarded-for",
  "x-vercel-proxied-for",
  "cf-connecting-ip",
  "true-client-ip",
  "x-client-ip",
  "forwarded",
]);
const PRIVATE_HEADER_PREFIX = "x-vercel-ip-";

function isPrivateHeader(name: string) {
  const header = name.toLowerCase();
  return (
    PRIVATE_HEADERS.has(header) || header.startsWith(PRIVATE_HEADER_PREFIX)
  );
}

// Keeps the request's body, cookies, credential headers and the visitor's IP
// address out of server and edge events.
export function stripRequestCredentials<T extends Event>(event: T): T {
  const { request } = event;
  if (!request) return event;
  delete request.data;
  delete request.cookies;
  if (!request.headers) return event;
  request.headers = Object.fromEntries(
    Object.entries(request.headers).filter(([name]) => !isPrivateHeader(name)),
  );
  return event;
}

const SENTRY_USER_FIELDS = new Set(["id", "role"]);

// Error monitoring runs without consent (legitimate interest), so a user is
// known to Sentry by their id and role only: never their email, name or IP
// address, whichever code put them on the scope.
export function stripUserPII<T extends Event>(event: T): T {
  if (!event.user) return event;
  const user = Object.fromEntries(
    Object.entries(event.user).filter(([field]) =>
      SENTRY_USER_FIELDS.has(field),
    ),
  );
  if (Object.keys(user).length) event.user = user;
  else delete event.user;
  return event;
}

// The SDK copies the scope's user id, email and username into every log's and
// span's attributes, web vital spans name the user by email in a bare `user`,
// and server spans carry the visitor's address both as attributes of their own
// (`http.client_ip`, `net.peer.ip`, `client.address`, `network.peer.address`)
// and in every request header (`http.request.header.x_forwarded_for`,
// `…x-real-ip`, `…x_vercel_ip_city`). Names are matched with dots and dashes
// read as underscores, so one fragment covers every spelling.
const IP_ATTRIBUTE_FRAGMENTS = [
  "forwarded",
  "proxied_for",
  "real_ip",
  "connecting_ip",
  "client_ip",
  "vercel_ip",
  "peer_ip",
  "peer_addr",
  "client_address",
  "ip_address",
];

function isIPAttribute(name: string) {
  const key = name.toLowerCase().replace(/[.-]/g, "_");
  if (key === "ip" || key.endsWith("_ip")) return true;
  return IP_ATTRIBUTE_FRAGMENTS.some((fragment) => key.includes(fragment));
}

function isPrivateAttribute(name: string) {
  if (name === "user") return true;
  if (name.startsWith("user.")) return name !== "user.id";
  return isIPAttribute(name);
}

function withoutPrivateAttributes<V>(attributes: Record<string, V>) {
  return Object.fromEntries(
    Object.entries(attributes).filter(([name]) => !isPrivateAttribute(name)),
  );
}

type SentryOptions = NonNullable<Parameters<typeof init>[0]>;
type SentryLog = Parameters<NonNullable<SentryOptions["beforeSendLog"]>>[0];
type SentrySpan = Parameters<NonNullable<SentryOptions["beforeSendSpan"]>>[0];

// Logs are known to Sentry by the user id only, like errors and transactions.
export function stripLogUserPII(log: SentryLog): SentryLog {
  if (!log.attributes) return log;
  log.attributes = withoutPrivateAttributes(log.attributes);
  return log;
}

// Transactions carry their root span's attributes in the trace context and
// every child span's in its own data; an error raised inside a span carries
// that span's attributes in its trace context too. beforeSend and
// beforeSendTransaction are the last places to take the visitor's IP
// address, email and name out of them.
export function stripSpanPII<T extends Event>(event: T): T {
  const trace = event.contexts?.trace;
  if (trace?.data) trace.data = withoutPrivateAttributes(trace.data);
  event.spans?.forEach(stripSpanDataPII);
  return event;
}

// Standalone spans, such as the browser's web vitals, skip
// beforeSendTransaction and pass through beforeSendSpan only.
export function stripSpanDataPII(span: SentrySpan): SentrySpan {
  if (span.data) span.data = withoutPrivateAttributes(span.data);
  return span;
}
