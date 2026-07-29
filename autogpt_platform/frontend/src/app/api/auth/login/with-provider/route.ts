import { auth } from "@/lib/auth/auth";
import { LoginProvider } from "@/types/auth";
import { APIError } from "better-auth/api";
import { NextResponse } from "next/server";
import { isWaitlistError, logWaitlistError } from "../../utils";

export async function POST(request: Request) {
  let body: { provider?: LoginProvider; redirectTo?: string };
  try {
    body = await request.json();
  } catch {
    // A malformed body is the caller's mistake, not ours — reporting 500 would
    // put client syntax errors in our server-error telemetry and invite retries.
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
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
        // Match on the body message ("Signups are not allowed."), not
        // error.message — the latter is the status ("FORBIDDEN"), which never
        // matches the waitlist patterns (same trap as the email signup path).
        if (isWaitlistError(error.body?.code, error.body?.message)) {
          logWaitlistError("OAuth Provider", error.message);
          return NextResponse.json({ error: "not_allowed" }, { status: 403 });
        }

        // Preserve Better Auth's status so callers can tell throttling (429)
        // or auth failure (401) apart from a bad request.
        return NextResponse.json(
          { error: error.body?.message || error.message },
          { status: error.statusCode ?? 400 },
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
