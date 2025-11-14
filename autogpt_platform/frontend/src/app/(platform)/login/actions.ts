"use server";

import BackendAPI from "@/lib/autogpt-server-api";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { environment } from "@/services/environment";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { shouldShowOnboarding } from "../../api/helpers";

export async function login(
  email: string,
  password: string,
  turnstileToken?: string,
) {
  try {
    const parsed = loginFormSchema.safeParse({ email, password });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid email or password",
      };
    }

    const captchaOk = await verifyTurnstileToken(turnstileToken ?? "", "login");
    if (!captchaOk && !environment.isVercelPreview()) {
      return {
        success: false,
        error: "CAPTCHA verification failed. Please try again.",
      };
    }

    const supabase = await getServerSupabase();
    if (!supabase) {
      return {
        success: false,
        error: "Authentication service unavailable",
      };
    }

    const { error } = await supabase.auth.signInWithPassword(parsed.data);
    if (error) {
      return {
        success: false,
        error: error.message,
      };
    }

    const api = new BackendAPI();
    await api.createUser();

    const onboarding = await shouldShowOnboarding();

    return {
      success: true,
      onboarding,
      next: onboarding ? "/onboarding" : "/",
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to login. Please try again.",
    };
  }
}
