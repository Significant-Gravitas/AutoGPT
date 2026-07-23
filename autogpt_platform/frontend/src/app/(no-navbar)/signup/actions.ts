"use server";

import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus } from "@/app/api/helpers";
import { auth } from "@/lib/auth/auth";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "@/services/analytics/datafast-server";
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
        // Match on the body message ("Signups are not allowed."), not
        // error.message — the latter is the status ("FORBIDDEN"), which never
        // matches the waitlist patterns, so rejections would slip through as a
        // generic error.
        if (isWaitlistError(error.body?.code, error.body?.message)) {
          logWaitlistError("Signup", error.message);
          return { success: false, error: "not_allowed" };
        }

        // Better Auth's email sign-up throws USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL;
        // accept the legacy code too in case the adapter version changes.
        if (
          error.body?.code === "USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL" ||
          error.body?.code === "USER_ALREADY_EXISTS"
        ) {
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
      const createUserResponse = await postV1GetOrCreateUser();
      if (wasAccountCreated(createUserResponse)) {
        await scheduleAccountCreatedGoal("email");
      }
    } catch (createUserError) {
      console.error("Error creating user during signup:", createUserError);
      Sentry.captureException(createUserError);
      // The session cookie is already set; revoke it so the browser's auth
      // state matches the failure the UI is about to show.
      await rollbackSession();
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
