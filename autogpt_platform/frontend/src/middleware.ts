import { updateSession } from "@/lib/supabase/middleware";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  // Redirect www to non-www to prevent cookie/auth domain mismatch (#9188)
  const host = request.headers.get("host") || "";
  if (host.startsWith("www.")) {
    const url = request.nextUrl.clone();
    url.host = host.replace(/^www\./, "");
    return NextResponse.redirect(url, 301);
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
