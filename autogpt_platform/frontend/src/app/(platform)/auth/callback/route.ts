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

// Handle the callback to complete the user session login
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");

  let next = "/marketplace";

  if (code) {
    const supabase = await getServerSupabase();

    if (!supabase) {
      return NextResponse.redirect(`${origin}/error`);
    }

    const { error } = await supabase.auth.exchangeCodeForSession(code);

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

        // Handle ApiError from the backend API client
        if (
          createUserError &&
          typeof createUserError === "object" &&
          "status" in createUserError
        ) {
          const apiError = createUserError as any;

          if (apiError.status === 401) {
            // Authentication issues - token missing/invalid
            return NextResponse.redirect(
              `${origin}/error?message=auth-token-invalid`,
            );
          } else if (apiError.status >= 500) {
            // Server/database errors
            return NextResponse.redirect(
              `${origin}/error?message=server-error`,
            );
          } else if (apiError.status === 429) {
            // Rate limiting
            return NextResponse.redirect(
              `${origin}/error?message=rate-limited`,
            );
          }
        }

        // Handle network/fetch errors
        if (
          createUserError instanceof TypeError &&
          createUserError.message.includes("fetch")
        ) {
          return NextResponse.redirect(`${origin}/error?message=network-error`);
        }

        // Generic user creation failure
        return NextResponse.redirect(
          `${origin}/error?message=user-creation-failed`,
        );
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
