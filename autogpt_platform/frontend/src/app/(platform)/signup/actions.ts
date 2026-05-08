"use server";

import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus, resolveResponse } from "@/app/api/helpers";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { signupFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { isWaitlistError, logWaitlistError } from "../../api/auth/utils";

export async function signup(
  email: string,
  password: string,
  confirmPassword: string,
  agreeToTerms: boolean,
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

    try {
      await resolveResponse(postV1GetOrCreateUser());
    } catch (createUserError) {
      console.error("Error creating user during signup:", createUserError);
      Sentry.captureException(createUserError);
      return {
        success: false,
        error: "Failed to complete account setup. Please try again.",
      };
    }

    const { shouldShowOnboarding } = await getOnboardingStatus();

    return {
      success: true,
      next: shouldShowOnboarding ? "/onboarding" : "/copilot",
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to sign up. Please try again.",
    };
  }
}
