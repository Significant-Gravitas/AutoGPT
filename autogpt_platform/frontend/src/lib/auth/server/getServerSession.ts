import { auth } from "@/lib/auth/auth";
import { isAPIError } from "better-auth/api";
import { headers } from "next/headers";
import { cache } from "react";

/**
 * Reads the Better Auth session for the current request and lets failures
 * through. `null` means the auth backend answered "no valid session" (missing,
 * expired or revoked); a throw means it couldn't answer (DB down, timeout).
 * Callers that decide whether to sign the user out need that difference.
 */
export const readServerSession = cache(async () => {
  return await auth.api.getSession({ headers: await headers() });
});

/**
 * Reads the Better Auth session for the current request. React-cached so
 * repeated calls within one request hit the DB (or cookie cache) once.
 */
export const getServerSession = cache(async () => {
  try {
    return await readServerSession();
  } catch (error) {
    console.error("Failed to read auth session:", error);
    return null;
  }
});

/** True when a session read failed because the session itself is invalid. */
export function isSessionRejection(error: unknown): boolean {
  return isAPIError(error) && error.statusCode === 401;
}
