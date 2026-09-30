/**
 * Client-side guards against OAuth open redirects on the consent page (#15048).
 *
 * Deny/Return and Approve must never navigate to an unregistered redirect_uri
 * or an opaque backend redirect_url that does not match a registered callback.
 */

export function isRegisteredRedirectUri(
  redirectUri: string | null | undefined,
  registeredUris: readonly string[] | null | undefined,
): boolean {
  if (!redirectUri || !registeredUris || registeredUris.length === 0) {
    return false;
  }
  return registeredUris.includes(redirectUri);
}

/**
 * Backend builds redirects as `{registered_redirect_uri}?{params}`.
 * Accept only when origin + pathname match the registered URI exactly.
 */
export function isSafeOAuthRedirectUrl(
  redirectUrl: string | null | undefined,
  registeredRedirectUri: string | null | undefined,
): boolean {
  if (!redirectUrl || !registeredRedirectUri) {
    return false;
  }
  try {
    const candidate = new URL(redirectUrl);
    const registered = new URL(registeredRedirectUri);
    return (
      candidate.origin === registered.origin &&
      candidate.pathname === registered.pathname
    );
  } catch {
    return false;
  }
}

export function buildOAuthAccessDeniedRedirect(
  redirectUri: string,
  state: string | null | undefined,
): string {
  const params = new URLSearchParams({
    error: "access_denied",
    error_description: "User denied access",
    state: state || "",
  });
  return `${redirectUri}?${params.toString()}`;
}
