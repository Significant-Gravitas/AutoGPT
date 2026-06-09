import { auth, toLegacyAuthUser } from "@/lib/auth/auth";
import { NextResponse, type NextRequest } from "next/server";
import { isAdminPage, isProtectedPage } from "./helpers";

export async function updateSession(request: NextRequest) {
  const response = NextResponse.next({ request });

  try {
    const session = await auth.api.getSession({
      headers: request.headers,
      query: {
        disableCookieCache: true,
      },
    });

    const user = session ? toLegacyAuthUser(session.user as never) : null;
    const pathname = request.nextUrl.pathname;

    if (!user && (isProtectedPage(pathname) || isAdminPage(pathname))) {
      const url = request.nextUrl.clone();
      const currentDest = url.pathname + url.search;
      url.pathname = "/login";
      url.search = `?next=${encodeURIComponent(currentDest)}`;
      return NextResponse.redirect(url);
    }

    if (user && user.role !== "admin" && isAdminPage(pathname)) {
      const url = request.nextUrl.clone();
      url.pathname = "/";
      url.search = "";
      return NextResponse.redirect(url);
    }
  } catch (error) {
    console.error("Failed to run Better Auth middleware", error);
  }

  return response;
}
