"use server";

import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { environment } from "@/services/environment";
import { signupFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { isWaitlistError, logWaitlistError } from "../../api/auth/utils";
import { shouldShowOnboarding } from "../../api/helpers";

export async function signup(
  email: string,
  password: string,
  confirmPassword: string,
  agreeToTerms: boolean,
  turnstileToken?: string,
) {
  try {
    const parsed = signupFormSchema.safeParse({
      email,
      password,
      confirmPassword,
      agreeToTerms,
    });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid signup payload",
      };
    }

    const captchaOk = await verifyTurnstileToken(
      turnstileToken ?? "",
      "signup",
    );

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

    const { data, error } = await supabase.auth.signUp(parsed.data);

    if (error) {
      if (isWaitlistError(error?.code, error?.message)) {
        logWaitlistError("Signup", error.message);
        return { success: false, error: "not_allowed" };
      }

      if ((error as any).code === "user_already_exists") {
        return { success: false, error: "user_already_exists" };
      }

      return {
        success: false,
        error: error.message,
      };
    }

    if (data.session) {
      await supabase.auth.setSession(data.session);
    }

    const isOnboardingEnabled = await shouldShowOnboarding();
    const next = isOnboardingEnabled ? "/onboarding" : "/";

    return { success: true, next };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to sign up. Please try again.",
    };
  }
}
