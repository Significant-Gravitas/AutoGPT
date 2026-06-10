import { getCookieCache, getSessionCookie } from "better-auth/cookies";
import { NextResponse, type NextRequest } from "next/server";
import { isAdminPage, isProtectedPage } from "./helpers";

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
    const cached = await getCookieCache(request, {
      secret: process.env.BETTER_AUTH_SECRET,
    });
    if (cached?.user) {
      return (cached.user as { role?: string }).role ?? null;
    }
  } catch {
    // fall through to the full session fetch
  }

  try {
    const response = await fetch(
      new URL("/api/auth/get-session", request.url),
      { headers: { cookie: request.headers.get("cookie") || "" } },
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
  if (!sessionCookie && hasLegacySupabaseSession(request)) {
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
