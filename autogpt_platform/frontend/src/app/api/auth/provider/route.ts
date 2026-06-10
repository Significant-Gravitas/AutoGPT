import { auth } from "@/lib/auth/auth";
import { LoginProvider } from "@/types/auth";
import { APIError } from "better-auth/api";
import { NextResponse } from "next/server";
import { isWaitlistError, logWaitlistError } from "../utils";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const provider: LoginProvider | undefined = body?.provider;
    const redirectTo: string | undefined = body?.redirectTo;

    if (!provider) {
      return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
    }

    try {
      const { url } = await auth.api.signInSocial({
        body: {
          provider,
          callbackURL:
            redirectTo || process.env.AUTH_CALLBACK_URL || "/auth/callback",
        },
        headers: request.headers,
      });

      return NextResponse.json({ url });
    } catch (error) {
      if (error instanceof APIError) {
        // Check for waitlist/allowlist error
        if (isWaitlistError(error.body?.code, error.message)) {
          logWaitlistError("OAuth Provider", error.message);
          return NextResponse.json({ error: "not_allowed" }, { status: 403 });
        }

        return NextResponse.json(
          { error: error.body?.message || error.message },
          { status: 400 },
        );
      }
      throw error;
    }
  } catch {
    return NextResponse.json(
      { error: "Failed to initiate OAuth" },
      { status: 500 },
    );
  }
}
