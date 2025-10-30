import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { NextResponse } from "next/server";
import { LoginProvider } from "@/types/auth";
import { isWaitlistError, logWaitlistError } from "../utils";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const provider: LoginProvider | undefined = body?.provider;
    const redirectTo: string | undefined = body?.redirectTo;

    if (!provider) {
      return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
    }

    const supabase = await getServerSupabase();
    if (!supabase) {
      return NextResponse.json(
        { error: "Authentication service unavailable" },
        { status: 500 },
      );
    }

    const { data, error } = await supabase.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo:
          redirectTo ||
          process.env.AUTH_CALLBACK_URL ||
          `http://localhost:3000/auth/callback`,
      },
    });

    if (error) {
      // Check for waitlist/allowlist error
      if (isWaitlistError(error?.code, error?.message)) {
        logWaitlistError("OAuth Provider", error.message);
        return NextResponse.json({ error: "not_allowed" }, { status: 403 });
      }

      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    return NextResponse.json({ url: data?.url });
  } catch {
    return NextResponse.json(
      { error: "Failed to initiate OAuth" },
      { status: 500 },
    );
  }
}
