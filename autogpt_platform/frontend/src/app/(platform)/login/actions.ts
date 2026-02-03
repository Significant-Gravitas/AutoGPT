"use server";

import BackendAPI from "@/lib/autogpt-server-api";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { getOnboardingStatus } from "../../api/helpers";

export async function login(email: string, password: string) {
  try {
    const parsed = loginFormSchema.safeParse({ email, password });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid email or password",
      };
    }

    const supabase = await getServerSupabase();
    if (!supabase) {
      return {
        success: false,
        error: "Authentication service unavailable",
      };
    }

    const { error } = await supabase.auth.signInWithPassword(parsed.data);
    if (error) {
      return {
        success: false,
        error: error.message,
      };
    }

    const api = new BackendAPI();
    await api.createUser();

    // Get onboarding status from backend (includes chat flag evaluated for this user)
    const { shouldShowOnboarding } = await getOnboardingStatus();
    const next = shouldShowOnboarding ? "/onboarding" : "/";

    return {
      success: true,
      next,
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to login. Please try again.",
    };
  }
}
