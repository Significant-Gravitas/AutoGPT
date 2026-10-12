"use server";

import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { getOnboardingStatus } from "@/app/api/helpers";
import { auth } from "@/lib/auth/auth";
import { getEmailVerificationCallbackURL } from "@/lib/auth/email-verification";
import { recordSignupConsent } from "@/lib/auth/server/recordSignupConsent";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import { markAccountCreated } from "@/services/analytics/account-created-server";
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
  marketingOptOut: boolean,
  next?: string | null,
) {
  try {
    const parsed = signupFormSchema.safeParse({
      email,
      password,
      confirmPassword,
      marketingOptOut,
    });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid signup payload",
      };
    }

    let signUpResult;
    try {
      // The session cookie is set automatically by the nextCookies plugin.
      signUpResult = await auth.api.signUpEmail({
        body: {
          email: parsed.data.email,
          password: parsed.data.password,
          name: parsed.data.email.split("@")[0],
          callbackURL: getEmailVerificationCallbackURL({
            next,
            marketingOptOut: parsed.data.marketingOptOut,
          }),
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

    // With email verification required there is no session yet: Better Auth
    // has emailed a link instead (and answers an address that already has an
    // account the same way). The platform user, the sign-up conversion and
    // the consent record wait for that link, which lands on /auth/callback.
    if (!signUpResult.token) {
      return {
        success: true,
        verificationRequired: true,
        email: parsed.data.email,
      };
    }

    try {
      const createUserResponse = await postV1GetOrCreateUser();
      if (wasAccountCreated(createUserResponse)) {
        await scheduleAccountCreatedGoal("email");
        await markAccountCreated("email");
        // Never throws, so a failed consent write can't reach the rollback
        // below: the account exists and the signup still succeeds.
        await recordSignupConsent({
          userID: signUpResult.user.id,
          marketingOptOut: parsed.data.marketingOptOut,
        });
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
      next: shouldShowOnboarding ? "/onboarding" : "/home",
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to sign up. Please try again.",
    };
  }
}
