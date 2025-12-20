"use server";

import { serverLogin } from "@/lib/auth/actions";
import BackendAPI from "@/lib/autogpt-server-api";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { shouldShowOnboarding } from "../../api/helpers";

export async function login(email: string, password: string) {
  try {
    const parsed = loginFormSchema.safeParse({ email, password });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid email or password",
      };
    }

    const result = await serverLogin({
      email: parsed.data.email,
      password: parsed.data.password,
    });

    if (result.error || !result.data) {
      return {
        success: false,
        error: result.error?.message || "Login failed",
      };
    }

    // Note: API calls may fail here because the auth cookies we just set
    // aren't available yet for the proxy route (it's a new HTTP request).
    // Default to showing onboarding if we can't check, and let the
    // onboarding flow handle user creation if needed.
    let onboarding = true;
    try {
      const api = new BackendAPI();
      await api.createUser();
      onboarding = await shouldShowOnboarding();
    } catch (error) {
      console.debug(
        "Could not complete post-login setup, defaulting to onboarding:",
        error,
      );
    }

    return {
      success: true,
      onboarding,
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to login. Please try again.",
    };
  }
}
