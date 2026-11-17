/**
 * The backend refused a send (a 4xx), so the email didn't go out. Any other
 * failure (a 5xx, the request cancelled, or lost after it left) leaves open
 * whether it went out. Its own module so tests that replace `email.ts` still
 * have it.
 */
export class AuthEmailNotSent extends Error {
  override name = "AuthEmailNotSent";
}

// By name, so a copy of this module (a reloaded test registry, a second
// bundle) still recognises it.
export function isAuthEmailNotSent(error: unknown) {
  return error instanceof Error && error.name === "AuthEmailNotSent";
}
