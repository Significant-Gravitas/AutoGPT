/**
 * Sanitize the `?next=` query parameter used by auth flows (`/login`, `/signup`)
 * to redirect users after authentication.
 *
 * Only accept same-origin relative paths starting with a single `/`. Reject
 * absolute URLs and protocol-relative paths (`//host`) so a crafted
 * `/login?next=https://phishing.site` cannot redirect users off-site.
 *
 * Backslashes are rejected outright: the WHATWG URL parser treats `\` as `/`
 * for http(s), so `new URL("/\\evil.com", origin)` resolves to
 * `https://evil.com/` — a prefix check alone lets that through. Callers hand
 * the result to `window.location.href` / `NextResponse.redirect`, which use
 * that same parser. Note the value arrives already percent-decoded (via
 * `searchParams.get`), so `%5C` is covered by the same check.
 *
 * Returns `null` when the value is missing, empty, or unsafe — callers should
 * fall back to their default destination in that case.
 */
export function sanitizeAuthNext(
  rawNext: string | null | undefined,
): string | null {
  if (!rawNext) return null;
  if (!rawNext.startsWith("/")) return null;
  if (rawNext.startsWith("//")) return null;
  if (rawNext.includes("\\")) return null;
  // The URL parser strips tab/CR/LF *before* parsing, so a tab-bearing value
  // could slip past the checks above and then resolve as "//evil.com".
  if (hasControlCharacter(rawNext)) return null;
  // /copilot is only a server redirect to /home; a client navigation into that
  // redirect from a (no-navbar) page crashes the App Router (React #310).
  return rawNext.replace(/^\/copilot\/?(?=[?#]|$)/, "/home");
}

const DEFAULT_LANDING_PATHS = new Set(["/", "/home", "/home/"]);
const URL_BASE = "http://localhost";
// Keys that only say where a visit came from. Home's own code reads none of
// them, but analytics reads them off the page an auth flow lands on, so
// withAuthTracking carries them there.
const TRACKING_PARAM =
  /^(utm_.*|gclid|gbraid|wbraid|fbclid|msclkid|ttclid|twclid|li_fat_id|mc_cid|mc_eid|ref|_gl)$/i;

// `/` (which drops its query) or `/home` with nothing but tracking params.
export function isDefaultLanding(next: string): boolean {
  const url = new URL(next, URL_BASE);
  if (!DEFAULT_LANDING_PATHS.has(url.pathname)) return false;
  if (url.pathname === "/") return true;
  if (url.hash) return false;
  return [...url.searchParams.keys()].every((key) => TRACKING_PARAM.test(key));
}

/**
 * The page an auth flow must send the user to, or `null` to route by
 * onboarding state. A default landing is `null` too, so a user who hasn't
 * finished onboarding is never sent to Home first. Every place that decides
 * whether a deep link is pending must use this, or one waits on a redirect
 * the other never makes.
 */
export function getAuthDeepLink(
  rawNext: string | null | undefined,
): string | null {
  const next = sanitizeAuthNext(rawNext);
  if (next === null || isDefaultLanding(next)) return null;
  return next;
}

/**
 * `destination` with the tracking params of a default-landing `next` added,
 * so a campaign or ad visit keeps its attribution when an auth flow routes
 * by onboarding state instead of following `next`.
 */
export function withAuthTracking(
  destination: string,
  rawNext: string | null | undefined,
): string {
  const next = sanitizeAuthNext(rawNext);
  if (next === null || !isDefaultLanding(next)) return destination;
  const url = new URL(destination, URL_BASE);
  new URL(next, URL_BASE).searchParams.forEach((value, key) => {
    if (TRACKING_PARAM.test(key)) url.searchParams.append(key, value);
  });
  return url.pathname + url.search + url.hash;
}

function hasControlCharacter(value: string) {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}
