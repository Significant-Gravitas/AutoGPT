import { authMiddleware } from "@/lib/auth/middleware";
import {
  isPostHogProxyPath,
  proxyPostHogRequest,
} from "@/providers/posthog/posthog-proxy-middleware";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  // PostHog's own requests never reach the auth checks: a redirect here would
  // silently drop events.
  if (isPostHogProxyPath(request.nextUrl.pathname)) {
    return proxyPostHogRequest(request) ?? NextResponse.next();
  }

  const canonicalURL = getCanonicalURL(request);
  if (canonicalURL) return NextResponse.redirect(canonicalURL, 308);

  return await authMiddleware(request);
}

function getCanonicalURL(request: NextRequest) {
  // A plain URL: a cloned NextURL puts a stripped trailing slash back.
  const url = new URL(request.nextUrl.href);
  let changed = false;

  // Redirect www to non-www so auth cookies are issued against a single,
  // canonical host and avoid the auth/cookie domain mismatch (#9188).
  // Use url.hostname (already lowercase-normalized by the URL parser) instead
  // of the raw Host header, which RFC 7230 treats as case-insensitive.
  if (url.hostname.startsWith("www.")) {
    url.hostname = url.hostname.slice(4);
    changed = true;
  }

  // Next's own trailing-slash redirect is off for the PostHog proxy's sake
  // (`skipTrailingSlashRedirect` in next.config.mjs), so pages keep it here.
  if (url.pathname !== "/" && url.pathname.endsWith("/")) {
    url.pathname = url.pathname.replace(/\/+$/, "") || "/";
    changed = true;
  }

  return changed ? url : null;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - /_next/static (static files)
     * - /_next/image (image optimization files)
     * - /favicon.ico (favicon file)
     * - /auth/callback (OAuth callback - needs to work without auth)
     * - /api/proxy (backend API proxy - the route handler authenticates
     *   itself via httpOnly cookies)
     * Feel free to modify this pattern to include more paths.
     *
     * Keep the PostHog proxy path (/relay) matched: middleware is what
     * forwards it to PostHog.
     *
     * Note: /auth/authorize and /auth/integrations/* ARE protected and need
     * middleware to run for authentication checks.
     */
    "/((?!_next/static|_next/image|favicon.ico|auth/callback|auth/integrations/mcp_callback|api/proxy|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
