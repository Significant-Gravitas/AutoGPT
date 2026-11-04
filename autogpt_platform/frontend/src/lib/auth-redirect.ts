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

// Where a signed-in user lands when nothing else was asked for. The middleware
// still sends a signed-out visit to `/` or `/home` through `/login?next=/home`,
// so such a `next` names no page the user chose (SECRT-3384).
const DEFAULT_LANDING_PATHS = new Set(["/", "/home", "/home/"]);

/**
 * True when `next` (already sanitized) only points at a default landing page:
 * `/` or `/home` (which `/copilot` maps to) with no query or hash. Anything
 * more specific, like `/home?sessionId=…`, is a real deep link.
 */
export function isDefaultLanding(next: string): boolean {
  const url = new URL(next, "http://localhost");
  return (
    DEFAULT_LANDING_PATHS.has(url.pathname) && url.search === "" && !url.hash
  );
}

/**
 * The page an auth flow must send the user to instead of its usual
 * destination, or `null` to route by onboarding state (`/onboarding` for a
 * user who has not finished it, else `/home`). A default landing is `null`
 * too: honouring `next=/home` would put a brand-new user on Home, with its
 * paywall and out-of-credits prompts, before onboarding.
 *
 * Every place that decides whether a deep link is pending (the auth pages,
 * `/auth/callback` and OnboardingProvider) must use this, or one of them
 * waits on a redirect the other never makes.
 */
export function getAuthDeepLink(
  rawNext: string | null | undefined,
): string | null {
  const next = sanitizeAuthNext(rawNext);
  if (next === null || isDefaultLanding(next)) return null;
  return next;
}

function hasControlCharacter(value: string) {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}
