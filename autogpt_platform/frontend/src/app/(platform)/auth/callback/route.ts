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

// Validate redirect URL to prevent open redirect attacks and malformed URLs
function validateRedirectUrl(url: string): string {
  try {
    // Clean up the URL first
    const cleanUrl = url.trim();

    // Check for malformed URL patterns that could cause issues
    if (
      cleanUrl.includes(",%20") ||
      cleanUrl.includes(", /") ||
      cleanUrl.includes(" /")
    ) {
      console.warn("Detected malformed redirect URL:", cleanUrl);
      return "/marketplace"; // Default to marketplace for malformed URLs
    }

    // Only allow relative URLs that start with /
    if (cleanUrl.startsWith("/") && !cleanUrl.startsWith("//")) {
      // Additional validation for common problematic patterns
      if (cleanUrl.includes("%20/") || cleanUrl.split("/").length > 10) {
        console.warn("Suspicious redirect URL pattern:", cleanUrl);
        return "/marketplace";
      }
      return cleanUrl;
    }

    // Default to marketplace for any invalid URLs
    return "/marketplace";
  } catch (error) {
    console.error("Error validating redirect URL:", error);
    return "/marketplace";
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
