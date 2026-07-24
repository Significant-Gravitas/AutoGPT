import { auth } from "@/lib/auth/auth";
import { cookies, headers } from "next/headers";
import { cache } from "react";

// JWT-per-session cache so every proxied backend call doesn't re-mint a token.
// TOKEN_EXPIRY_MARGIN_MS refreshes the cached token 5 minutes before it
// actually expires: it must stay comfortably under the 1h JWT lifetime (see
// `jwt.expirationTime` in auth.ts) while still absorbing clock skew between
// this process and the backend that validates the token.
const TOKEN_EXPIRY_MARGIN_MS = 5 * 60 * 1000;
// Bounds cache memory: a JWT is ~1KB, so 1000 entries ≈ ~1MB worst case. The
// Map keeps insertion order, so the oldest entry is evicted first when full.
const MAX_TOKEN_CACHE_ENTRIES = 1000;
const serverTokenCache = new Map<
  string,
  { token: string; expiresAt: number }
>();

export function readJwtExpiryMs(token: string): number {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split(".")[1], "base64url").toString("utf-8"),
    );
    if (typeof payload.exp === "number") return payload.exp * 1000;
  } catch {
    // Only reached for a malformed/opaque token — Better Auth JWTs always
    // carry a numeric `exp`. Defensive fallback so a bad decode can't cache a
    // token effectively forever.
  }
  return Date.now() + TOKEN_EXPIRY_MARGIN_MS * 2;
}

export function cacheServerToken(sessionCookie: string, token: string): void {
  if (serverTokenCache.size >= MAX_TOKEN_CACHE_ENTRIES) {
    const oldestKey = serverTokenCache.keys().next().value;
    if (oldestKey) serverTokenCache.delete(oldestKey);
  }
  serverTokenCache.set(sessionCookie, {
    token,
    expiresAt: readJwtExpiryMs(token) - TOKEN_EXPIRY_MARGIN_MS,
  });
}

export function getCachedServerToken(sessionCookie: string): string | null {
  const cached = serverTokenCache.get(sessionCookie);
  if (cached && cached.expiresAt > Date.now()) return cached.token;
  return null;
}

/**
 * Mints (or returns a cached) backend-API JWT for the current request's
 * session by calling the Better Auth server instance IN-PROCESS via
 * `auth.api.getToken`. The Python backend validates the JWT against
 * /api/auth/jwks.
 *
 * This deliberately replaces the previous HTTP self-fetch to
 * `/api/auth/token`: under constrained preview concurrency that fan-out could
 * deadlock (every worker awaiting another worker's token request), which hung
 * the Copilot page forever. An in-process call has no such dependency.
 *
 * Server-only: it statically imports `auth`, which pulls in `pg`. It must
 * never execute in the browser, so client-graph modules (custom-mutator, the
 * makeAuthenticated* helpers) reach it via a runtime-guarded dynamic import.
 */
export const getServerAuthToken = cache(async (): Promise<string | null> => {
  try {
    const cookieStore = await cookies();
    const sessionCookie = cookieStore
      .getAll()
      .find(
        ({ name }) =>
          name === "better-auth.session_token" ||
          name === "__Secure-better-auth.session_token",
      );
    if (!sessionCookie) return null;

    const cached = getCachedServerToken(sessionCookie.value);
    if (cached) return cached;

    const { token } = await auth.api.getToken({ headers: await headers() });
    if (!token) return null;

    cacheServerToken(sessionCookie.value, token);
    return token;
  } catch (error) {
    console.error("Failed to get auth token:", error);
    return null;
  }
});
