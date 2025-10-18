import { NextResponse } from "next/server";
import * as Sentry from "@sentry/nextjs";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { signupFormSchema } from "@/types/auth";
import { shouldShowOnboarding } from "../../helpers";
import { isWaitlistError, logWaitlistError } from "../utils";

export async function POST(request: Request) {
  console.log("=== SIGNUP ROUTE START ===");
  console.log("Timestamp:", new Date().toISOString());

  try {
    const body = await request.json();
    console.log("Request body received:", {
      hasEmail: !!body?.email,
      hasPassword: !!body?.password,
      hasConfirmPassword: !!body?.confirmPassword,
      agreeToTerms: body?.agreeToTerms,
      hasTurnstileToken: !!body?.turnstileToken,
    });

    const parsed = signupFormSchema.safeParse({
      email: body?.email,
      password: body?.password,
      confirmPassword: body?.confirmPassword,
      agreeToTerms: body?.agreeToTerms,
    });

    if (!parsed.success) {
      console.error("Schema validation failed:", parsed.error.errors);
      return NextResponse.json(
        { error: "Invalid signup payload" },
        { status: 400 },
      );
    }

    console.log("Schema validation passed for email:", parsed.data.email);

    const turnstileToken: string | undefined = body?.turnstileToken;

    console.log("Starting CAPTCHA verification...");
    const captchaOk = await verifyTurnstileToken(
      turnstileToken ?? "",
      "signup",
    );

    if (!captchaOk) {
      console.error("CAPTCHA verification failed");
      return NextResponse.json(
        { error: "CAPTCHA verification failed. Please try again." },
        { status: 400 },
      );
    }
    console.log("CAPTCHA verification successful");

    console.log("Getting Supabase client...");
    const supabase = await getServerSupabase();
    if (!supabase) {
      console.error("Failed to get Supabase client");
      return NextResponse.json(
        { error: "Authentication service unavailable" },
        { status: 500 },
      );
    }
    console.log("Supabase client obtained successfully");

    console.log("Attempting signup for email:", parsed.data.email);
    const { data, error } = await supabase.auth.signUp(parsed.data);
    console.log("Supabase signup response:", {
      hasData: !!data,
      hasError: !!error,
      hasSession: !!data?.session,
      hasUser: !!data?.user,
    });

    if (error) {
      console.error("=== SIGNUP ERROR DETAILS ===");
      console.error("Error object:", {
        message: error.message,
        code: (error as any).code,
        status: (error as any).status,
        name: error.name,
        stack: error.stack?.split("\n")[0], // First line of stack
      });

      // Check for waitlist/allowlist error
      const isWaitlist = isWaitlistError(error);
      console.log("Is waitlist error?", isWaitlist);

      if (isWaitlist) {
        console.log(">>> WAITLIST ERROR DETECTED <<<");
        console.log("Error message before sanitization:", error.message);
        logWaitlistError("Signup", error.message);
        return NextResponse.json({ error: "not_allowed" }, { status: 403 });
      }

      if ((error as any).code === "user_already_exists") {
        console.log("User already exists error");
        return NextResponse.json(
          { error: "user_already_exists" },
          { status: 409 },
        );
      }

      console.log("Other signup error:", error.message);
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    if (data.session) {
      console.log("Setting session for new user");
      await supabase.auth.setSession(data.session);
    } else {
      console.log("No session returned - user may need email confirmation");
    }

    const isOnboardingEnabled = await shouldShowOnboarding();
    const next = isOnboardingEnabled ? "/onboarding" : "/";
    console.log("Signup successful. Redirecting to:", next);
    console.log("=== SIGNUP ROUTE END SUCCESS ===");

    return NextResponse.json({ success: true, next });
  } catch (err) {
    console.error("=== UNEXPECTED ERROR IN SIGNUP ===");
    console.error("Error:", err);
    Sentry.captureException(err);
    return NextResponse.json(
      { error: "Failed to sign up. Please try again." },
      { status: 500 },
    );
  }
}
