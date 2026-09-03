import { getCookieCache, getSessionCookie } from "better-auth/cookies";
import { NextResponse, type NextRequest } from "next/server";
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
  // REL-001 cache coherence: the cookieCache (better-auth.session_data,
  // maxAge 5m) is a signed but *stale* snapshot — after logout or
  // revokeSessionsOnPasswordReset the session row is deleted but the
  // cache cookie survives until expiry/clear. Trusting it directly lets
  // a revoked session pass the admin gate for up to 5m. Backend JWT
  // revocation (Redis revoked:sid/jti, 5m TTL, fail-open bounded by
  // expiry) is the hard gate for API access; Edge cannot query Redis.
  // For the admin gate we therefore treat the cache as a hint:
  // - if the cache says non-admin, that is safe to return quickly;
  // - if it says admin (or is missing), we verify via the DB-backed
  //   /api/auth/get-session fetch below, which correctly 401s on a
  //   revoked/deleted session. This keeps the happy path cheap while
  //   closing the 5m stale-cache window for privilege escalation.
  try {
    const cached = await getCookieCache(request, {
      secret: process.env.BETTER_AUTH_SECRET,
    });
    if (cached?.user) {
      const cachedRole = (cached.user as { role?: string }).role ?? null;
      if (cachedRole !== "admin") return cachedRole;
      // cached admin → fall through to DB verification so a revoked
      // admin session does not keep admin access via stale cookie.
    } else if (cached) {
      // cached = null would have thrown; no user means fall through
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
        headers: { cookie: request.headers.get("cookie") || "" },
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

/**
 * Route-protection middleware.
 *
 * Follows the Better Auth guidance for Next.js: the middleware only does an
 * optimistic cookie-presence check for protected pages; real session
 * validation happens in route handlers and server components. Admin pages
 * additionally resolve the user's role (cookie cache first, then a session
 * fetch).
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
