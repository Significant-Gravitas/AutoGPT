import { NextResponse } from "next/server";
import * as Sentry from "@sentry/nextjs";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { signupFormSchema } from "@/types/auth";
import { shouldShowOnboarding } from "../../helpers";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const parsed = signupFormSchema.safeParse({
      email: body?.email,
      password: body?.password,
      confirmPassword: body?.confirmPassword,
      agreeToTerms: body?.agreeToTerms,
    });

    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid signup payload" },
        { status: 400 },
      );
    }

    const turnstileToken: string | undefined = body?.turnstileToken;

    const captchaOk = await verifyTurnstileToken(
      turnstileToken ?? "",
      "signup",
    );
    if (!captchaOk) {
      return NextResponse.json(
        { error: "CAPTCHA verification failed. Please try again." },
        { status: 400 },
      );
    }

    const supabase = await getServerSupabase();
    if (!supabase) {
      return NextResponse.json(
        { error: "Authentication service unavailable" },
        { status: 500 },
      );
    }

    const { data, error } = await supabase.auth.signUp(parsed.data);

    if (error) {
      // FIXME: supabase doesn't return the correct error message for this case
      if (error.message.includes("P0001")) {
        return NextResponse.json({ error: "not_allowed" }, { status: 403 });
      }
      if ((error as any).code === "user_already_exists") {
        return NextResponse.json(
          { error: "user_already_exists" },
          { status: 409 },
        );
      }
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    if (data.session) {
      await supabase.auth.setSession(data.session);
    }

    const isOnboardingEnabled = await shouldShowOnboarding();
    const next = isOnboardingEnabled ? "/onboarding" : "/";

    return NextResponse.json({ success: true, next });
  } catch (err) {
    Sentry.captureException(err);
    return NextResponse.json(
      { error: "Failed to sign up. Please try again." },
      { status: 500 },
    );
  }
}
