"use server";

import { auth } from "@/lib/auth/auth";
import {
  EMAIL_NOT_VERIFIED_CODE,
  getEmailVerificationCallbackURL,
} from "@/lib/auth/email-verification";
import { rollbackSession } from "@/lib/auth/server/rollbackSession";
import BackendAPI from "@/lib/autogpt-server-api";
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

    try {
      await auth.api.signInEmail({
        body: {
          email: parsed.data.email,
          password: parsed.data.password,
          // Only used for the link Better Auth emails to an unverified user.
          callbackURL: getEmailVerificationCallbackURL(next),
        },
        headers: await headers(),
      });
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

    try {
      const api = new BackendAPI();
      await api.createUser();
    } catch (createUserError) {
      // The session cookie is already set; revoke it so the browser's auth
      // state matches the failure the UI is about to show.
      await rollbackSession();
      throw createUserError;
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
      error: "Failed to login. Please try again.",
    };
  }
}
