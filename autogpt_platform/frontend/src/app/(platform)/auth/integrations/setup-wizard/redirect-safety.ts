/**
 * Client-side guards against OAuth open redirects on the integration setup
 * wizard (#15303).
 *
 * Continue and Cancel must never navigate to a `redirect_uri` the query string
 * can name freely. The sibling consent page (`auth/authorize`) gates the same
 * navigation on the app's registered callbacks; this module is the wizard's
 * copy of that rule, kept next to its own page so it does not depend on the
 * authorize page's module landing first.
 *
 * Two rules, applied together:
 *
 * 1. The target must be an `http(s)` URL. A `javascript:` value assigned to
 *    `window.location.href` runs in the platform's origin with the victim's
 *    session, so it is rejected before any registration check runs.
 * 2. When the wizard knows which app is asking, the target must be one of that
 *    app's registered redirect URIs, compared exactly. Comparing origins is not
 *    enough: custom-scheme callbacks such as `myapp://callback` have an opaque
 *    `"null"` origin.
 *
 * Rule 2 needs the app's registered callbacks from the backend, which
 * `GET /oauth/app/{client_id}` does not expose yet — #15057 adds the field to
 * that response for the authorize page. Until it lands, an unknown app has no
 * list to compare against and rule 1 is all that can be enforced, so a caller
 * without a `client_id` can still be sent to any `https` host. Sending
 * `client_id` is what closes that: the check below starts enforcing
 * registration the moment the field appears, with no further change here.
 */

/**
 * The registered callbacks the OAuth app info carries, read defensively.
 *
 * `OAuthApplicationPublicInfo` does not declare `redirect_uris` yet (#15057
 * adds it), so an app whose info predates that field simply has no list and
 * falls back to the scheme rule above.
 */
export function registeredRedirectUrisOf(appInfo: unknown): readonly string[] {
  const uris = (appInfo as { redirect_uris?: unknown } | null | undefined)
    ?.redirect_uris;
  if (!Array.isArray(uris)) {
    return [];
  }
  return uris.filter((uri): uri is string => typeof uri === "string");
}

/** Parses `value`, or returns null when it is not a URL at all. */
function parseUrl(value: string | null | undefined): URL | null {
  if (!value) return null;
  try {
    return new URL(value);
  } catch {
    return null;
  }
}

/**
 * True when `value` is a URL the browser can navigate to over http(s).
 *
 * `javascript:`, `data:`, `blob:` and custom schemes are rejected here: the
 * WHATWG URL parser accepts them, and `window.location.href` acts on them.
 */
export function isHttpRedirectUrl(value: string | null | undefined): boolean {
  const parsed = parseUrl(value);
  if (!parsed) return false;
  return parsed.protocol === "http:" || parsed.protocol === "https:";
}

/**
 * True when `redirectUri` is one of the app's registered callbacks.
 *
 * An empty list means the app is unknown (no `client_id`, or the backend does
 * not expose the field yet), so there is nothing to compare against and the
 * caller can only rely on the scheme rule.
 */
export function isRegisteredRedirectUri(
  redirectUri: string | null | undefined,
  registeredUris: readonly string[] | null | undefined,
): boolean {
  if (!registeredUris || registeredUris.length === 0) {
    return true;
  }
  return registeredUris.includes(redirectUri as string);
}

/**
 * Appends the wizard's result parameters to a redirect target.
 *
 * Built with `URL.searchParams` rather than string interpolation so a
 * registered target that already carries a query string (or a fragment) keeps
 * it intact, and so nothing in the parameters can be read as part of the URL.
 */
export function buildSetupWizardRedirect(
  redirectUri: string,
  params: Record<string, string>,
): string {
  const url = new URL(redirectUri);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}
