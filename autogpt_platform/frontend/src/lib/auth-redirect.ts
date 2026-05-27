/**
 * Sanitize the `?next=` query parameter used by auth flows (`/login`, `/signup`)
 * to redirect users after authentication.
 *
 * Only accept same-origin relative paths starting with a single `/`. Reject
 * absolute URLs and protocol-relative paths (`//host`) so a crafted
 * `/login?next=https://phishing.site` cannot redirect users off-site.
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
  return rawNext;
}
