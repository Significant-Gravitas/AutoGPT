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
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001F\u007F]/.test(rawNext)) return null;
  return rawNext;
}
