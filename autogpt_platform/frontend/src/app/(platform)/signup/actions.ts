"use server";

import { serverRegister } from "@/lib/auth/actions";
import { isWaitlistError } from "@/lib/auth";
import { signupFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { shouldShowOnboarding } from "../../api/helpers";

function logWaitlistError(context: string, message: string) {
  console.log(`[${context}] Waitlist error: ${message}`);
}

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

    const result = await serverRegister({
      email: parsed.data.email,
      password: parsed.data.password,
    });

    if (result.error || !result.data) {
      const error = result.error;

      if (error && isWaitlistError(error)) {
        logWaitlistError("Signup", error.message);
        return { success: false, error: "not_allowed" };
      }

      if (error?.message?.includes("already registered")) {
        return { success: false, error: "user_already_exists" };
      }

      return {
        success: false,
        error: error?.message || "Registration failed",
      };
    }

    // Note: shouldShowOnboarding may fail here because the auth cookies we just set
    // aren't available yet for the proxy route (it's a new HTTP request).
    // Default to showing onboarding for new users, which is the safer path.
    let isOnboardingEnabled = true;
    try {
      isOnboardingEnabled = await shouldShowOnboarding();
    } catch (error) {
      console.debug(
        "Could not check onboarding status after signup, defaulting to onboarding:",
        error,
      );
    }
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
