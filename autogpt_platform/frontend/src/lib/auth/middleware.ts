/**
 * Middleware utilities for authentication.
 *
 * Used by Next.js middleware to protect routes and manage sessions.
 */

import { NextRequest, NextResponse } from "next/server";

import { AUTH_COOKIE_NAMES } from "./constants";
import { PROTECTED_PAGES, ADMIN_PAGES, isAdmin } from "./helpers";
import { decodeJwtPayload } from "./api";

/**
 * Check if a path should skip authentication middleware.
 */
export function shouldSkipAuth(pathname: string): boolean {
  // Skip static files and Next.js internals
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/static") ||
    pathname.startsWith("/api") ||
    pathname.includes(".") // Files with extensions (images, etc.)
  ) {
    return true;
  }

  // Skip auth-related pages
  if (
    pathname === "/login" ||
    pathname === "/signup" ||
    pathname === "/reset-password" ||
    pathname.startsWith("/auth/")
  ) {
    return true;
  }

  return false;
}

/**
 * Check if a path requires authentication.
 */
export function requiresAuth(pathname: string): boolean {
  return PROTECTED_PAGES.some((page) => pathname.startsWith(page));
}

/**
 * Check if a path requires admin access.
 */
export function requiresAdmin(pathname: string): boolean {
  return ADMIN_PAGES.some((page) => pathname.startsWith(page));
}

/**
 * Handle authentication in middleware.
 * Returns a response if redirect is needed, otherwise undefined.
 */
export function handleAuthMiddleware(
  request: NextRequest,
): NextResponse | undefined {
  const { pathname } = request.nextUrl;

  // Skip if path doesn't need auth handling
  if (shouldSkipAuth(pathname)) {
    return undefined;
  }

  // Get access token from cookies
  const accessToken = request.cookies.get(
    AUTH_COOKIE_NAMES.ACCESS_TOKEN,
  )?.value;
  const refreshToken = request.cookies.get(
    AUTH_COOKIE_NAMES.REFRESH_TOKEN,
  )?.value;

  const hasSession = Boolean(accessToken || refreshToken);

  // Check if path requires authentication
  if (requiresAuth(pathname)) {
    if (!hasSession) {
      // Redirect to login with return URL
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("redirect", pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  // Check if path requires admin access
  if (requiresAdmin(pathname)) {
    if (!hasSession) {
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("redirect", pathname);
      return NextResponse.redirect(loginUrl);
    }

    // Check admin role from token
    if (accessToken) {
      const user = decodeJwtPayload(accessToken);
      if (!user || !isAdmin(user.role)) {
        // Redirect non-admins to marketplace
        const marketplaceUrl = new URL("/marketplace", request.url);
        return NextResponse.redirect(marketplaceUrl);
      }
    }
  }

  // Redirect logged-in users away from login/signup pages
  if (hasSession && (pathname === "/login" || pathname === "/signup")) {
    const marketplaceUrl = new URL("/marketplace", request.url);
    return NextResponse.redirect(marketplaceUrl);
  }

  return undefined;
}
