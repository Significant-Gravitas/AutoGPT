"use server";

import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus, resolveResponse } from "@/app/api/helpers";
import { auth } from "@/lib/auth/auth";
import { signupFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { APIError } from "better-auth/api";
import { headers } from "next/headers";
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

    try {
      // The session cookie is set automatically by the nextCookies plugin.
      await auth.api.signUpEmail({
        body: {
          email: parsed.data.email,
          password: parsed.data.password,
          name: parsed.data.email.split("@")[0],
        },
        headers: await headers(),
      });
    } catch (error) {
      if (error instanceof APIError) {
        if (isWaitlistError(error.body?.code, error.message)) {
          logWaitlistError("Signup", error.message);
          return { success: false, error: "not_allowed" };
        }

        if (error.body?.code === "USER_ALREADY_EXISTS") {
          return { success: false, error: "user_already_exists" };
        }

        return {
          success: false,
          error: error.body?.message || error.message,
        };
      }
      throw error;
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
