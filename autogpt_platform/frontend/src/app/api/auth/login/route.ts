import BackendAPI from "@/lib/autogpt-server-api";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import { NextResponse } from "next/server";
import { shouldShowOnboarding } from "../../helpers";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const parsed = loginFormSchema.safeParse({
      email: body?.email,
      password: body?.password,
    });

    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid email or password" },
        { status: 400 },
      );
    }

    const turnstileToken: string | undefined = body?.turnstileToken;

    // Verify Turnstile token if provided
    const captchaOk = await verifyTurnstileToken(turnstileToken ?? "", "login");
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

    const { error } = await supabase.auth.signInWithPassword(parsed.data);
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    const api = new BackendAPI();
    await api.createUser();

    const onboarding = await shouldShowOnboarding();

    return NextResponse.json({
      success: true,
      onboarding,
      next: onboarding ? "/onboarding" : "/",
    });
  } catch (err) {
    Sentry.captureException(err);
    return NextResponse.json(
      { error: "Failed to login. Please try again." },
      { status: 500 },
    );
  }
}
