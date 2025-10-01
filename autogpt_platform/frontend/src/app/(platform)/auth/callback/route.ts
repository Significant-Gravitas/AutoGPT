import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import BackendAPI from "@/lib/autogpt-server-api";
import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

async function shouldShowOnboarding() {
  const api = new BackendAPI();
  return (
    (await api.isOnboardingEnabled()) &&
    !(await api.getUserOnboarding()).completedSteps.includes("CONGRATS")
  );
}

// Default redirect path - matches the home page redirect destination
const DEFAULT_REDIRECT_PATH = "/marketplace";

// Validate redirect URL to prevent open redirect attacks and malformed URLs
function validateRedirectUrl(url: string): string {
  try {
    const cleanUrl = url.trim();

    // Check for completely invalid patterns that suggest URL corruption
    if (
      cleanUrl.includes(",") || // Any comma suggests concatenated URLs
      cleanUrl.includes(" ") // Spaces in URLs are problematic
    ) {
      console.warn(
        "Detected corrupted redirect URL (likely race condition):",
        cleanUrl,
      );
      return DEFAULT_REDIRECT_PATH;
    }

    // Only allow relative URLs that start with /
    if (!cleanUrl.startsWith("/") || cleanUrl.startsWith("//")) {
      console.warn("Invalid redirect URL format:", cleanUrl);
      return DEFAULT_REDIRECT_PATH;
    }

    // Additional safety checks
    if (cleanUrl.split("/").length > 5) {
      // Reasonable path depth limit
      console.warn("Suspiciously deep redirect URL:", cleanUrl);
      return DEFAULT_REDIRECT_PATH;
    }

    // For now, allow any valid relative path (can be restricted later if needed)
    return cleanUrl;
  } catch (error) {
    console.error("Error validating redirect URL:", error);
    return DEFAULT_REDIRECT_PATH;
  }
}

// Handle the callback to complete the user session login
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");

  // if "next" is in param, use it as the redirect URL
  const nextParam = searchParams.get("next") ?? "/";
  // Validate redirect URL to prevent open redirect attacks
  let next = validateRedirectUrl(nextParam);

  if (code) {
    const supabase = await getServerSupabase();

    if (!supabase) {
      return NextResponse.redirect(`${origin}/error`);
    }

    const { error } = await supabase.auth.exchangeCodeForSession(code);
    // data.session?.refresh_token is available if you need to store it for later use
    if (!error) {
      try {
        const api = new BackendAPI();
        await api.createUser();

        if (await shouldShowOnboarding()) {
          next = "/onboarding";
          revalidatePath("/onboarding", "layout");
        } else {
          revalidatePath("/", "layout");
        }
      } catch (createUserError) {
        console.error("Error creating user:", createUserError);
        // Continue with redirect even if createUser fails
      }

      const forwardedHost = request.headers.get("x-forwarded-host"); // original origin before load balancer
      const isLocalEnv = process.env.NODE_ENV === "development";
      if (isLocalEnv) {
        // we can be sure that there is no load balancer in between, so no need to watch for X-Forwarded-Host
        return NextResponse.redirect(`${origin}${next}`);
      } else if (forwardedHost) {
        return NextResponse.redirect(`https://${forwardedHost}${next}`);
      } else {
        return NextResponse.redirect(`${origin}${next}`);
      }
    }
  }

  // return the user to an error page with instructions
  return NextResponse.redirect(`${origin}/auth/auth-code-error`);
}
