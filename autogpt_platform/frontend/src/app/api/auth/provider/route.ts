import { getGoogleLoginUrl, isWaitlistError } from "@/lib/auth";
import { NextResponse } from "next/server";
import { LoginProvider } from "@/types/auth";

function logWaitlistError(context: string, message: string) {
  console.log(`[${context}] Waitlist error: ${message}`);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const provider: LoginProvider | undefined = body?.provider;

    if (!provider) {
      return NextResponse.json({ error: "Invalid provider" }, { status: 400 });
    }

    // Currently only Google OAuth is supported
    if (provider !== "google") {
      return NextResponse.json(
        { error: "Provider not supported" },
        { status: 400 },
      );
    }

    const result = await getGoogleLoginUrl();

    if (result.error) {
      // Check for waitlist/allowlist error
      if (isWaitlistError(result.error)) {
        logWaitlistError("OAuth Provider", result.error.message);
        return NextResponse.json({ error: "not_allowed" }, { status: 403 });
      }

      return NextResponse.json(
        { error: result.error.message },
        { status: 400 },
      );
    }

    return NextResponse.json({ url: result.data?.url });
  } catch {
    return NextResponse.json(
      { error: "Failed to initiate OAuth" },
      { status: 500 },
    );
  }
}
