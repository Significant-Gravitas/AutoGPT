import {
  getCookieCache,
  getSessionCookie,
  parseCookies,
  SECURE_COOKIE_PREFIX,
} from "better-auth/cookies";
import { constantTimeEqual, makeSignature } from "better-auth/crypto";
import { NextResponse, type NextRequest } from "next/server";
import { usesSecureCookies } from "./base-url";
import { IP_ADDRESS_HEADERS } from "./client-ip";
import { isAdminPage, isProtectedPage } from "./helpers";
import { canConsumeLegacyCookies } from "./legacy-cookies";

const SUPABASE_AUTH_COOKIE = /^sb-.+-auth-token(\.\d+)?$/;

function hasLegacySupabaseSession(request: NextRequest): boolean {
  return request.cookies
    .getAll()
    .some(({ name }) => SUPABASE_AUTH_COOKIE.test(name));
}

async function getSessionUserRole(
  request: NextRequest,
): Promise<string | null> {
  // Fast path: the signed cookie cache carries the user without a DB hit.
  try {
    const cached = await getOwnCookieCache(request);
    if (cached) {
      return (cached.user as { role?: string }).role ?? null;
    }
  } catch {
    // fall through to the full session fetch
  }

  try {
    // Bounded so a stalled session lookup can't hang navigation. On any error
    // (including this timeout) we fall through to null = "not admin", the safe
    // default. Edge runtime here has no pg access, so this stays an HTTP call
    // rather than the in-process auth API used elsewhere.
    const response = await fetch(
      new URL("/api/auth/get-session", request.url),
      {
        headers: sessionLookupHeaders(request),
        signal: AbortSignal.timeout(3000),
      },
    );
    if (!response.ok) return null;
    const session = await response.json();
    return session?.user?.role ?? null;
  } catch {
    return null;
  }
}

// The signed cookie cache, if it belongs to the request's session. As Better
// Auth's own /get-session checks, the session token inside the cache must be
// the one in the request's signed session cookie.
async function getOwnCookieCache(request: NextRequest) {
  const secret = process.env.BETTER_AUTH_SECRET;
  if (!secret) return null;
  // Under the names auth.ts gives the cookies. Left to guess, Better Auth
  // looks for the `__Secure-` names in any production build, which a server
  // whose base URL is http never sets (#17308).
  const isSecure = usesSecureCookies();
  const token = await getSignedSessionToken({ request, secret, isSecure });
  if (!token) return null;
  const cached = await getCookieCache(request, { secret, isSecure });
  return cached?.session.token === token ? cached : null;
}

interface SignedCookieArgs {
  request: NextRequest;
  secret: string;
  isSecure: boolean;
}

// The token in the request's session cookie, if its signature is valid. The
// cookie holds `<token>.<signature>`: better-call's signCookieValue signs the
// token with this same makeSignature (HMAC-SHA256, base64).
async function getSignedSessionToken({
  request,
  secret,
  isSecure,
}: SignedCookieArgs) {
  const name = `${isSecure ? SECURE_COOKIE_PREFIX : ""}better-auth.session_token`;
  const value = parseCookies(request.headers.get("cookie") ?? "").get(name);
  const separator = value?.lastIndexOf(".") ?? -1;
  if (!value || separator < 1) return null;
  const token = value.slice(0, separator);
  const signature = value.slice(separator + 1);
  const expected = await makeSignature(token, secret);
  return constantTimeEqual(signature, expected) ? token : null;
}

// The visitor's cookies, and the headers Better Auth reads their IP address
// from, as they arrived. Better Auth then counts the lookup against the
// visitor, by the same rules as their own requests to /api/auth, rather than
// putting every visitor's lookups in one shared rate-limit bucket.
function sessionLookupHeaders(request: NextRequest) {
  const headers = new Headers({ cookie: request.headers.get("cookie") || "" });
  for (const name of IP_ADDRESS_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

/**
 * Route-protection middleware.
 *
 * Follows the Better Auth guidance for Next.js: the middleware only does an
 * optimistic cookie-presence check for protected pages; real session
 * validation happens in route handlers and server components. Admin pages
 * additionally resolve the user's role (the session's own cookie cache first,
 * then a session fetch).
 */
export async function authMiddleware(request: NextRequest) {
  const url = request.nextUrl.clone();
  const pathname = request.nextUrl.pathname;

  // API routes authenticate themselves.
  if (pathname.startsWith("/api/")) {
    return NextResponse.next();
  }

  const sessionCookie = getSessionCookie(request);

  // A logged-in Supabase user from before the auth migration: upgrade their
  // legacy session into a Better Auth session. The bridge endpoint clears the
  // Supabase cookies either way, so this runs at most once per browser.
  // Gated on the bridge actually being able to consume the cookies — without
  // SUPABASE_JWT_SECRET it bounces to /login with the cookies intact, and
  // this redirect would send them right back: an infinite loop. With the
  // secret unset the cookies are simply ignored (kept for a later bridge)
  // and the user takes the normal login path.
  if (
    !sessionCookie &&
    hasLegacySupabaseSession(request) &&
    canConsumeLegacyCookies()
  ) {
    const next = encodeURIComponent(url.pathname + url.search);
    url.pathname = "/api/auth/supabase-bridge";
    url.search = `?next=${next}`;
    return NextResponse.redirect(url);
  }

  if (!sessionCookie && (isProtectedPage(pathname) || isAdminPage(pathname))) {
    const currentDest = url.pathname + url.search;
    url.pathname = "/login";
    url.search = `?next=${encodeURIComponent(currentDest)}`;
    return NextResponse.redirect(url);
  }

  if (sessionCookie && isAdminPage(pathname)) {
    const role = await getSessionUserRole(request);
    if (role !== "admin") {
      url.pathname = "/";
      url.search = "";
      return NextResponse.redirect(url);
    }
  }

  return NextResponse.next();
}
