import { updateSession } from "@/lib/supabase/middleware";
import { type NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { match } from "@formatjs/intl-localematcher";
import Negotiator from "negotiator";
import { LOCALES, DEFAULT_LOCALE } from "@/lib/utils";

function getLocale(request: NextRequest) {
  try {
    let headers = Object.fromEntries(request.headers.entries());
    let languages = new Negotiator({ headers }).languages();
    if (languages.length === 0 || languages[0] === '*') {
      languages = ["en"];
    }
    return match(languages, LOCALES, DEFAULT_LOCALE);
  } catch (error) {
    console.error("Error getting locale, defaulting to English:", error);
    return "en";
  }
}

export async function middleware(request: NextRequest) {
  // Check if there is any supported locale in the pathname
  const { pathname } = request.nextUrl;
  const pathnameHasLocale = LOCALES.some(
    (locale) => pathname.startsWith(`/${locale}/`) || pathname === `/${locale}`,
  );
  let locale = "";
  if (!pathnameHasLocale) {
    // Redirect if there is no locale
    locale = `/${getLocale(request)}`;
  }

  return await updateSession(request, locale);
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * Feel free to modify this pattern to include more paths.
     */
    "/((?!_next/static|_next/image|favicon.ico|auth|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
