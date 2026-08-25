import * as Sentry from "@sentry/nextjs";
import { cookies } from "next/headers";
import {
  ACCOUNT_CREATED_COOKIE,
  ACCOUNT_CREATED_COOKIE_MAX_AGE_SECONDS,
  type SignupMethod,
} from "./account-created-cookie";

// Flags a brand-new account for the browser, which reports the Google Ads
// sign-up conversion on the next page it renders (AdsConversionTracker).
// Best effort: a tracking failure must never fail the signup itself.
export async function markAccountCreated(method: SignupMethod): Promise<void> {
  try {
    const cookieStore = await cookies();
    cookieStore.set(ACCOUNT_CREATED_COOKIE, method, {
      path: "/",
      maxAge: ACCOUNT_CREATED_COOKIE_MAX_AGE_SECONDS,
      sameSite: "lax",
      httpOnly: false,
    });
  } catch (error) {
    Sentry.captureException(error, {
      tags: { analytics_provider: "google_ads", analytics_goal: "signup" },
    });
  }
}
