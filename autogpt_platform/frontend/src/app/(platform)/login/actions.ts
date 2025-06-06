"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import * as Sentry from "@sentry/nextjs";
import getServerSupabase from "@/lib/supabase/getServerSupabase";
import BackendAPI from "@/lib/autogpt-server-api";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { verifyTurnstileToken } from "@/lib/turnstile";

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
    const supabase = getServerSupabase();
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
      console.error("Error logging in:", error);
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
      const supabase = getServerSupabase();
      const api = new BackendAPI();

      if (!supabase) {
        redirect("/error");
      }

      const { error } = await supabase!.auth.signInWithOAuth({
        provider: provider,
        options: {
          redirectTo:
            process.env.AUTH_CALLBACK_URL ??
            `http://localhost:3000/auth/callback`,
        },
      });

      if (error) {
        console.error("Error logging in", error);
        return error.message;
      }

      await api.createUser();
      // Don't onboard if disabled or already onboarded
      if (await shouldShowOnboarding()) {
        revalidatePath("/onboarding", "layout");
        redirect("/onboarding");
      }
    },
  );
}
