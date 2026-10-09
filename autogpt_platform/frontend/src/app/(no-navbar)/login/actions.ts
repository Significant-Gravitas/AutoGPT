"use server";

import { postV1GetOrCreateUser } from "@/app/api/__generated__/endpoints/auth/auth";
import { auth } from "@/lib/auth/auth";
import {
  EMAIL_NOT_VERIFIED_CODE,
  getEmailVerificationCallbackURL,
} from "@/lib/auth/email-verification";
import { recordSignupConsent } from "@/lib/auth/server/recordSignupConsent";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import { markAccountCreated } from "@/services/analytics/account-created-server";
import {
  scheduleAccountCreatedGoal,
  wasAccountCreated,
} from "@/services/analytics/datafast-server";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { APIError } from "better-auth/api";
import { headers } from "next/headers";
import { getOnboardingStatus } from "../../api/helpers";

export async function login(
  email: string,
  password: string,
  next?: string | null,
) {
  try {
    const parsed = loginFormSchema.safeParse({ email, password });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid email or password",
      };
    }

    let userID: string;
    try {
      const signInResult = await auth.api.signInEmail({
        body: {
          email: parsed.data.email,
          password: parsed.data.password,
          // Only used for the link Better Auth emails to an unverified user.
          callbackURL: getEmailVerificationCallbackURL({ next }),
        },
        headers: await headers(),
      });
      userID = signInResult.user.id;
    } catch (error) {
      if (error instanceof APIError) {
        // Right password, unverified address: Better Auth has just emailed a
        // fresh verification link (sendOnSignIn).
        if (error.body?.code === EMAIL_NOT_VERIFIED_CODE) {
          return {
            success: false,
            error: "email_not_verified",
            email: parsed.data.email,
          };
        }
        return {
          success: false,
          error: error.body?.message || "Invalid email or password",
        };
      }
      throw error;
    }

    // With verification required, this can be the call that creates the
    // account: a mail scanner may have opened the verify link first, so the
    // owner's own click lands here instead of on /auth/callback. The backend
    // reports a creation once, so the conversion can't count twice.
    let accountCreated = false;
    try {
      accountCreated = wasAccountCreated(await postV1GetOrCreateUser());
    } catch (createUserError) {
      // The session cookie is already set; revoke it so the browser's auth
      // state matches the failure the UI is about to show.
      await rollbackSession();
      throw createUserError;
    }
    if (accountCreated) {
      await scheduleAccountCreatedGoal("email");
      await markAccountCreated("email");
      // The account was made on /signup, under its legal line, so the terms
      // are recorded here. A refusal made there rode in the verification link
      // and can't be recovered at login (SECRT-2851). Never throws.
      await recordSignupConsent({ userID, marketingOptOut: false });
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
      error: "Failed to login. Please try again.",
    };
  }
}
