import { getServerSupabase } from "@/lib/auth/server/getServerSupabase";
import { NextResponse } from "next/server";
import { LOGIN_PROVIDERS, LoginProvider } from "@/types/auth";
import { isWaitlistError, logWaitlistError } from "../utils";

function isProviderConfigured(provider: LoginProvider) {
  switch (provider) {
    case "google":
      return Boolean(
        process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET,
      );
    case "github":
      return Boolean(
        process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET,
      );
    case "discord":
      return Boolean(
        process.env.DISCORD_CLIENT_ID && process.env.DISCORD_CLIENT_SECRET,
      );
    default:
      return false;
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const provider: LoginProvider | undefined = body?.provider;
    const redirectTo: string | undefined = body?.redirectTo;

    if (!provider || !LOGIN_PROVIDERS.includes(provider)) {
      return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
    }

    if (!isProviderConfigured(provider)) {
      return NextResponse.json(
        { error: `${provider} login is not configured` },
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
