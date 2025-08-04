"use server";
import BackendAPI from "@/lib/autogpt-server-api";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";

async function shouldShowOnboarding() {
  const api = new BackendAPI();
  return (
    (await api.isOnboardingEnabled()) &&
    !(await api.getUserOnboarding()).completedSteps.includes("CONGRATS")
  );
}

export async function login(
  values: z.infer<typeof loginFormSchema>,
  turnstileToken: string,
) {
  return await Sentry.withServerActionInstrumentation("login", {}, async () => {
    const supabase = await getServerSupabase();
    const api = new BackendAPI();

    if (!supabase) {
      redirect("/error");
    }

    // Verify Turnstile token if provided
    const success = await verifyTurnstileToken(turnstileToken, "login");
    if (!success) {
      return "CAPTCHA verification failed. Please try again.";
    }

    // We are sure that the values are of the correct type because zod validates the form
    const { error } = await supabase.auth.signInWithPassword(values);

    if (error) {
      return error.message;
    }

    await api.createUser();

    // Don't onboard if disabled or already onboarded
    if (await shouldShowOnboarding()) {
      revalidatePath("/onboarding", "layout");
      redirect("/onboarding");
    }

    revalidatePath("/", "layout");
    redirect("/");
  });
}

export async function providerLogin(provider: LoginProvider) {
  return await Sentry.withServerActionInstrumentation(
    "providerLogin",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        redirect("/error");
      }

      const { data, error } = await supabase!.auth.signInWithOAuth({
        provider: provider,
        options: {
          redirectTo:
            process.env.AUTH_CALLBACK_URL ??
            `http://localhost:3000/auth/callback`,
        },
      });

      if (error) {
        // FIXME: supabase doesn't return the correct error message for this case
        if (error.message.includes("P0001")) {
          return "not_allowed";
        }

        console.error("Error logging in", error);
        return error.message;
      }

      // Redirect to the OAuth provider's URL
      if (data?.url) {
        redirect(data.url);
      }

      // Note: api.createUser() and onboarding check happen in the callback handler
      // after the session is established. See `auth/callback/route.ts`.
    },
  );
}
