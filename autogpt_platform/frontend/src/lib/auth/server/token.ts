import { auth } from "@/lib/auth/auth";
import { headers } from "next/headers";

const SESSION_COOKIE_PATTERN =
  /(?:^|;\s*)(?:__Secure-)?better-auth\.session_token=([^;]+)/;

// JWT-per-session cache so every proxied backend call doesn't re-sign a
// token. Entries expire 5 minutes before the JWT itself does.
const EXPIRY_MARGIN_MS = 5 * 60 * 1000;
const MAX_CACHE_ENTRIES = 1000;
const tokenCache = new Map<string, { token: string; expiresAt: number }>();

function readJwtExpiryMs(token: string): number {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split(".")[1], "base64url").toString("utf-8"),
    );
    if (typeof payload.exp === "number") return payload.exp * 1000;
  } catch {
    // fall through to a conservative default
  }
  return Date.now() + EXPIRY_MARGIN_MS * 2;
}

/**
 * Mints (or returns a cached) backend-API JWT for the current request's
 * session. The Python backend validates it against /api/auth/jwks.
 */
export async function getBackendAuthToken(): Promise<string | null> {
  const requestHeaders = await headers();
  const cookieHeader = requestHeaders.get("cookie") || "";
  const sessionCookie = cookieHeader.match(SESSION_COOKIE_PATTERN)?.[1];

  if (!sessionCookie) return null;

  const cached = tokenCache.get(sessionCookie);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.token;
  }

  try {
    const result = await auth.api.getToken({ headers: requestHeaders });
    const token = result?.token;
    if (!token) return null;

    if (tokenCache.size >= MAX_CACHE_ENTRIES) {
      const oldestKey = tokenCache.keys().next().value;
      if (oldestKey) tokenCache.delete(oldestKey);
    }
    tokenCache.set(sessionCookie, {
      token,
      expiresAt: readJwtExpiryMs(token) - EXPIRY_MARGIN_MS,
    });

    return token;
  } catch (error) {
    console.error("Failed to mint backend auth token:", error);
    return null;
  }
}
