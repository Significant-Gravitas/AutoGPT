import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus, resolveResponse } from "@/app/api/helpers";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "@/services/analytics/datafast-server";
import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

// Handle the callback to complete the user session login
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");

  let next = "/copilot";

  if (code) {
    const supabase = await getServerSupabase();

    if (!supabase) {
      return NextResponse.redirect(`${origin}/error`);
    }

    const { error } = await supabase.auth.exchangeCodeForSession(code);

    if (!error) {
      try {
        const createUserResponse = await postV1GetOrCreateUser();
        await resolveResponse(Promise.resolve(createUserResponse));
        if (wasAccountCreated(createUserResponse.headers)) {
          await scheduleAccountCreatedGoal("google");
        }

        const { shouldShowOnboarding } = await getOnboardingStatus();
        next = shouldShowOnboarding ? "/onboarding" : "/copilot";
        revalidatePath(next, "layout");
      } catch (createUserError) {
        console.error("Error creating user:", createUserError);

        const errorStatus = getErrorStatus(createUserError);
        if (errorStatus === 401) {
          return NextResponse.redirect(
            `${origin}/error?message=auth-token-invalid`,
          );
        } else if (errorStatus !== null && errorStatus >= 500) {
          return NextResponse.redirect(`${origin}/error?message=server-error`);
        } else if (errorStatus === 429) {
          return NextResponse.redirect(`${origin}/error?message=rate-limited`);
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

      // Get redirect destination from 'next' query parameter
      next = searchParams.get("next") || next;

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

function getErrorStatus(error: unknown): number | null {
  if (
    !error ||
    typeof error !== "object" ||
    !("status" in error) ||
    typeof error.status !== "number"
  ) {
    return null;
  }

  return error.status;
}
