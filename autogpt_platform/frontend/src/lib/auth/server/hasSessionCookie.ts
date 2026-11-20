import { getSessionCookie } from "better-auth/cookies";
import { headers } from "next/headers";

/**
 * Whether the request carries a Better Auth session cookie. Without one the
 * visitor is signed out; with one they may be signed in, or the session may
 * have expired or been revoked.
 */
export async function hasSessionCookie() {
  return getSessionCookie(await headers()) !== null;
}
