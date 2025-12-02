import { environment } from "@/services/environment";
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { getCookieSettings, isAdminPage, isProtectedPage } from "./helpers";

export async function updateSession(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  });

  const supabaseUrl = environment.getSupabaseUrl();
  const supabaseKey = environment.getSupabaseAnonKey();
  const isAvailable = Boolean(supabaseUrl && supabaseKey);

  if (!isAvailable) {
    return supabaseResponse;
  }

  try {
    const supabase = createServerClient(supabaseUrl, supabaseKey, {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value),
          );
          supabaseResponse = NextResponse.next({
            request,
          });
          cookiesToSet.forEach(({ name, value, options }) => {
            supabaseResponse.cookies.set(name, value, {
              ...options,
              ...getCookieSettings(),
            });
          });
        },
      },
    });

    const userResponse = await supabase.auth.getUser();
    const user = userResponse.data.user;
    const userRole = user?.role;

    const url = request.nextUrl.clone();
    const pathname = request.nextUrl.pathname;

    // IMPORTANT: Avoid writing any logic between createServerClient and
    // supabase.auth.getUser(). A simple mistake could make it very hard to debug
    // issues with users being randomly logged out.

    // AUTH REDIRECTS
    // 1. Check if user is not authenticated but trying to access protected content
    if (!user) {
      const attemptingProtectedPage = isProtectedPage(pathname);
      const attemptingAdminPage = isAdminPage(pathname);

      if (attemptingProtectedPage || attemptingAdminPage) {
        url.pathname = "/login";
        return NextResponse.redirect(url);
      }
    }

    // 2. Check if user is authenticated but lacks admin role when accessing admin pages
    if (user && userRole !== "admin" && isAdminPage(pathname)) {
      url.pathname = "/marketplace";
      return NextResponse.redirect(url);
    }

    // IMPORTANT: You *must* return the supabaseResponse object as it is. If you're
    // creating a new response object with NextResponse.next() make sure to:
    // 1. Pass the request in it, like so:
    //    const myNewResponse = NextResponse.next({ request })
    // 2. Copy over the cookies, like so:
    //    myNewResponse.cookies.setAll(supabaseResponse.cookies.getAll())
    // 3. Change the myNewResponse object to fit your needs, but avoid changing
    //    the cookies!
    // 4. Finally:
    //    return myNewResponse
    // If this is not done, you may be causing the browser and server to go out
    // of sync and terminate the user's session prematurely!
  } catch (error) {
    console.error("Failed to run Supabase middleware", error);
  }

  return supabaseResponse;
}
