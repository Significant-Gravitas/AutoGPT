import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { shouldShowOnboarding } from "@/app/api/helpers";
import { setAuthCookies } from "@/lib/auth/cookies";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006";

// Handle the OAuth callback from the backend
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  const state = searchParams.get("state");

  let next = "/marketplace";

  if (code) {
    try {
      // Exchange the code with the backend's Google OAuth callback
      const callbackUrl = new URL(`${API_BASE_URL}/api/auth/google/callback`);
      callbackUrl.searchParams.set("code", code);
      if (state) {
        callbackUrl.searchParams.set("state", state);
      }

      const response = await fetch(callbackUrl.toString());
      const data = await response.json();

      if (!response.ok) {
        console.error("OAuth callback error:", data);
        return NextResponse.redirect(`${origin}/auth/auth-code-error`);
      }

      // Set the auth cookies with the tokens from the backend
      if (data.access_token && data.refresh_token) {
        await setAuthCookies(
          data.access_token,
          data.refresh_token,
          data.expires_in || 900, // Default 15 minutes
        );

        // Check if onboarding is needed
        if (await shouldShowOnboarding()) {
          next = "/onboarding";
          revalidatePath("/onboarding", "layout");
        } else {
          revalidatePath("/", "layout");
        }

        const forwardedHost = request.headers.get("x-forwarded-host");
        const isLocalEnv = process.env.NODE_ENV === "development";

        if (isLocalEnv) {
          return NextResponse.redirect(`${origin}${next}`);
        } else if (forwardedHost) {
          return NextResponse.redirect(`https://${forwardedHost}${next}`);
        } else {
          return NextResponse.redirect(`${origin}${next}`);
        }
      }
    } catch (error) {
      console.error("OAuth callback error:", error);
      return NextResponse.redirect(`${origin}/auth/auth-code-error`);
    }
  }

  // return the user to an error page with instructions
  return NextResponse.redirect(`${origin}/auth/auth-code-error`);
}
