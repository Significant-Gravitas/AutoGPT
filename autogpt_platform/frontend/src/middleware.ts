import { updateSession } from "@/lib/supabase/middleware";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  // Redirect www to non-www so Supabase cookies are issued against a single,
  // canonical host and avoid the auth/cookie domain mismatch (#9188).
  // Use url.hostname (already lowercase-normalized by the URL parser) instead
  // of the raw Host header, which RFC 7230 treats as case-insensitive.
  const url = request.nextUrl.clone();
  if (url.hostname.startsWith("www.")) {
    url.hostname = url.hostname.slice(4);
    return NextResponse.redirect(url, 308);
  }

  return await updateSession(request);
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - /_next/static (static files)
     * - /_next/image (image optimization files)
     * - /favicon.ico (favicon file)
     * - /auth/callback (OAuth callback - needs to work without auth)
     * Feel free to modify this pattern to include more paths.
     *
     * Note: /auth/authorize and /auth/integrations/* ARE protected and need
     * middleware to run for authentication checks.
     */
    "/((?!_next/static|_next/image|favicon.ico|auth/callback|auth/integrations/mcp_callback|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
