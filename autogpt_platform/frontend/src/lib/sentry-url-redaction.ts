import type {
  Breadcrumb,
  Event,
  init,
  replayIntegration,
} from "@sentry/nextjs";

// The backend's marker and rule (backend/util/url_redaction.py): every query
// value is masked and its key kept, and so is every credential path segment.
export const REDACTED = "[REDACTED]";

/**
 * Mask the credentials in a URL. A fragment is treated like a query, since in
 * the browser it can hold an `access_token=` too.
 */
export function redactURL(url: string): string {
  const start = url.search(/[?#]/);
  if (start < 0) return redactPath(url);
  return redactPath(url.slice(0, start) + redactQuery(url.slice(start)));
}

/** Mask the credentials in every URL inside a span description or message. */
export function redactURLsInText(text: string): string {
  return redactPath(text.replace(QUERY_IN_TEXT, redactQuery));
}

/**
 * True when a page's address carries a credential: in a path segment the
 * redactor masks, under a credential key in its query or fragment, or in the
 * address a redirect key points on to.
 */
export function holdsCredential(url: string): boolean {
  const start = url.search(/[?#]/);
  const path = start < 0 ? url : url.slice(0, start);
  if (redactPath(path) !== path) return true;
  if (start < 0) return false;
  return url
    .slice(start + 1)
    .split(/[?#&]/)
    .some((pair) => {
      const [key, ...value] = pair.split("=");
      const name = key.toLowerCase();
      if (CREDENTIAL_QUERY_KEYS.has(name)) return true;
      return (
        REDIRECT_QUERY_KEYS.has(name) &&
        holdsCredential(decode(value.join("=")))
      );
    });
}

type SentryOptions = NonNullable<Parameters<typeof init>[0]>;
type SentrySpan = Parameters<NonNullable<SentryOptions["beforeSendSpan"]>>[0];

/**
 * An event carries its page as `request.url`, the page before it as Referer,
 * earlier pages and calls in breadcrumbs and spans, and a replay's in `urls`.
 */
export function redactEventURLs<T extends Event & { urls?: string[] }>(
  event: T,
): T {
  redactRequestURLs(event.request);
  // An error is named after the page's raw path when its route isn't known.
  if (event.transaction) event.transaction = redactURL(event.transaction);
  if (event.urls) event.urls = event.urls.map(redactURL);
  event.breadcrumbs?.forEach(redactBreadcrumbURLs);
  const trace = event.contexts?.trace;
  if (trace?.data) redactAttributeURLs(trace.data);
  event.spans?.forEach(redactSpanURLs);
  return event;
}

export function redactSpanURLs<T extends SentrySpan>(span: T): T {
  if (span.description) span.description = redactURLsInText(span.description);
  if (span.data) redactAttributeURLs(span.data);
  return span;
}

// Navigation breadcrumbs record `from` and `to`; fetch, xhr and the replay's
// own click and hydration breadcrumbs record `url`.
const BREADCRUMB_URL_FIELDS = ["url", "from", "to"];

export function redactBreadcrumbURLs(breadcrumb: Breadcrumb): Breadcrumb {
  const { data } = breadcrumb;
  if (!data) return breadcrumb;
  for (const field of BREADCRUMB_URL_FIELDS) {
    if (typeof data[field] === "string") data[field] = redactURL(data[field]);
  }
  return breadcrumb;
}

type ReplayOptions = NonNullable<Parameters<typeof replayIntegration>[0]>;
type RecordingEvent = Parameters<
  NonNullable<ReplayOptions["beforeAddRecordingEvent"]>
>[0];

/**
 * A replay records breadcrumbs, page navigations and resource and network
 * timings as events of its own, each named by its URL.
 */
export function redactRecordingEventURLs(event: RecordingEvent) {
  if (event.data.tag === "breadcrumb") redactBreadcrumbURLs(event.data.payload);
  if (event.data.tag === "performanceSpan") {
    const span = event.data.payload;
    span.description = redactURL(span.description);
    const { data } = span;
    if (data && "previous" in data && typeof data.previous === "string") {
      data.previous = redactURL(data.previous);
    }
  }
  return event;
}

// Where fetch, xhr and Next's server spans, the semantic conventions and web
// vitals put a URL.
const URL_ATTRIBUTES = new Set([
  "url",
  "url.full",
  "url.path",
  "url.query",
  "url.fragment",
  "http.url",
  "http.target",
  "http.query",
  "http.fragment",
  "lcp.url",
  "transaction",
]);

function redactAttributeURLs(attributes: Record<string, unknown>) {
  for (const [name, value] of Object.entries(attributes)) {
    if (typeof value !== "string") continue;
    if (name.startsWith(HEADER_ATTRIBUTE_PREFIX)) {
      attributes[name] = redactHeader(name, value);
    } else if (QUERY_ATTRIBUTES.has(name)) {
      // The semantic conventions give a query or fragment without its ? or #.
      attributes[name] = redactQuery(value);
    } else if (URL_ATTRIBUTES.has(name)) {
      attributes[name] = redactURL(value);
    }
  }
}

function redactRequestURLs(request: Event["request"]) {
  if (!request) return;
  if (request.url) request.url = redactURL(request.url);
  // Server events also carry the parsed query, which can't be masked by key.
  if (request.query_string) request.query_string = REDACTED;
  const { headers } = request;
  if (!headers) return;
  for (const [name, value] of Object.entries(headers)) {
    headers[name] = redactHeader(name, value);
  }
}

// Any request header is read as a URL, since the Referer and Next's next-url
// name the page. Next's router state tree names it in a form no rule can parse.
function redactHeader(name: string, value: string) {
  return ROUTER_STATE_HEADER.test(name) ? REDACTED : redactURL(value);
}

// The backend's credential routes, matched anywhere (behind /api/proxy too) and
// zero-width so stacked prefixes all match; /api/invitations/pending is kept.
const API_CREDENTIAL_SEGMENT =
  /(?=(\/api\/(?:integrations\/(?:[^/?\s]+\/)?webhooks|platform-linking\/(?:user-)?tokens|public\/shared(?:\/chats)?|invitations(?!\/pending(?:[/?\s"']|$)))\/)([^/?\s]+))/g;
// Pages whose own path is the token, at the start of a URL or after a space.
// /link/slack is static, and a route name like /link/[token] holds no token.
const PAGE_CREDENTIAL_SEGMENT =
  /(^|\s)((?:[a-z][a-z\d+.-]*:)?\/\/[^/?#\s]*)?(\/(?:link\/(?!slack(?:[/?#\s]|$))|share\/(?!chat\/\[)(?:chat\/)?))(?!\[)[^/?\s]+/gi;
// Query keys that carry a credential into a page; a new page that receives one
// in its query adds its key here. Masking needs no list: it masks every value.
const CREDENTIAL_QUERY_KEYS = new Set([
  "token",
  "code",
  "state",
  "access_token",
  "refresh_token",
  "id_token",
]);
// Query keys whose value is the address a page sends the visitor on to.
const REDIRECT_QUERY_KEYS = new Set([
  "next",
  "callbackurl",
  "redirect_uri",
  "returnto",
]);
const QUERY_ATTRIBUTES = new Set([
  "url.query",
  "url.fragment",
  "http.query",
  "http.fragment",
]);
const HEADER_ATTRIBUTE_PREFIX = "http.request.header.";
const ROUTER_STATE_HEADER = /next[-_]router[-_]state[-_]tree$/i;
const QUERY_IN_TEXT = /\?\S+/g;
const QUERY_VALUE = /=[^&]*/g;

function redactQuery(query: string) {
  return query.replace(QUERY_VALUE, `=${REDACTED}`);
}

function decode(value: string) {
  const encoded = value.replace(/\+/g, " ");
  try {
    return decodeURIComponent(encoded);
  } catch {
    return encoded;
  }
}

// Built from the spans, as the backend does, so overlapping matches merge.
function redactPath(text: string) {
  let redacted = "";
  let done = 0;
  for (const match of text.matchAll(API_CREDENTIAL_SEGMENT)) {
    const start = match.index + match[1].length;
    const end = start + match[2].length;
    if (end <= done) continue;
    redacted += text.slice(done, Math.max(start, done)) + REDACTED;
    done = end;
  }
  return (redacted + text.slice(done)).replace(
    PAGE_CREDENTIAL_SEGMENT,
    `$1$2$3${REDACTED}`,
  );
}
