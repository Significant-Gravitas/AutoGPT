import * as Sentry from "@sentry/nextjs";
import { auth } from "@/lib/auth/auth";
import { isSessionRejection } from "@/lib/auth/server/getServerSession";
import { AuthTokenUnavailableError } from "@/lib/autogpt-server-api/helpers";
import { cookies, headers } from "next/headers";
import { cache } from "react";

/**
 * Mints a backend-API JWT for the current request's session by calling the
 * Better Auth server instance IN-PROCESS via `auth.api.getToken`. The Python
 * backend validates the JWT against /api/auth/jwks.
 *
 * This deliberately replaces the previous HTTP self-fetch to
 * `/api/auth/token`: under constrained preview concurrency that fan-out could
 * deadlock (every worker awaiting another worker's token request), which hung
 * the Copilot page forever. An in-process call has no such dependency.
 *
 * Only React's `cache()` memoizes it — deliberately per-request, not a
 * cross-request Map keyed on the session cookie. Such a cache returns a token
 * without re-checking the session, which silently outlives revocation: a stolen
 * cookie would keep minting backend access for the rest of the JWT's lifetime
 * even after logout or `revokeSessionsOnPasswordReset` deleted the session row.
 * Minting is in-process, so re-checking every request is cheap.
 *
 * Every failure propagates from here; getServerAuthToken decides which of
 * them mean there is no session to send.
 *
 * Server-only: it statically imports `auth`, which pulls in `pg`, so it must
 * never reach the browser bundle. Client-graph modules cannot import it at all
 * — not even dynamically, since webpack resolves dynamic imports in the same
 * layer; they keep the self-fetching variant in autogpt-server-api/helpers.ts.
 */
export const readServerAuthToken = cache(async (): Promise<string | null> => {
  const cookieStore = await cookies();
  const sessionCookie = cookieStore
    .getAll()
    .find(
      ({ name }) =>
        name === "better-auth.session_token" ||
        name === "__Secure-better-auth.session_token",
    );
  if (!sessionCookie) return null;

  const { token } = await auth.api.getToken({ headers: await headers() });
  return token ?? null;
});

/**
 * Returns null only when there is no session to send. A session whose token
 * cannot be minted (the auth database refused or timed out) throws
 * AuthTokenUnavailableError: sending that request on without a token would
 * get a 401 from the backend, which the browser reads as a logout. Better Auth
 * still reports a failed *session* lookup as a 401, so only failures after it
 * (the signing-key read, where the 5-minute session cookie cache sends most
 * requests) can be told apart.
 */
export const getServerAuthToken = cache(async (): Promise<string | null> => {
  try {
    await cookies();
  } catch (error) {
    // Outside a request (prerendering) there is no session to send.
    console.error("Failed to get auth token:", error);
    return null;
  }

  try {
    return await readServerAuthToken();
  } catch (error) {
    if (isSessionRejection(error)) return null;
    console.error("Failed to get auth token:", error);
    Sentry.captureException(error);
    throw new AuthTokenUnavailableError({ cause: error });
  }
});
