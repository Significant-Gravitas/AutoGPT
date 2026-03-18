"use server";

import {
  postV1CheckIfAnEmailIsAllowedToSignUp,
  postV1GetOrCreateUser,
} from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus, resolveResponse } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
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

    // Pre-check invite eligibility before creating a Supabase auth user.
    // This prevents orphaned auth accounts when the invite gate is enabled.
    try {
      const checkResult = await resolveResponse(
        postV1CheckIfAnEmailIsAllowedToSignUp({ email: parsed.data.email }),
      );
      if (!checkResult.allowed) {
        return { success: false, error: "not_allowed" };
      }
      // If the check fails (non-OK or backend unreachable), fall through to
      // signup — the backend-level check in get_or_activate_user() catches it.
    } catch (precheckError) {
      if (precheckError instanceof ApiError) {
        Sentry.captureMessage(
          `Invite pre-check returned HTTP ${precheckError.status}`,
          { level: "warning", tags: { flow: "signup_precheck" } },
        );
      } else {
        Sentry.captureException(precheckError, {
          tags: { flow: "signup_precheck" },
        });
      }
      // Graceful fallback: don't block signup if the pre-check itself fails.
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

    // Get onboarding status from backend (includes chat flag evaluated for this user)
    const { shouldShowOnboarding } = await getOnboardingStatus();
    const next = shouldShowOnboarding ? "/onboarding" : "/";

    return { success: true, next };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to sign up. Please try again.",
    };
  }
}
