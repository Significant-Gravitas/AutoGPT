import { exchangePasswordResetCode } from "@/lib/supabase/helpers";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const code = searchParams.get("code");
  const origin =
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";

  if (!code) {
    return NextResponse.redirect(
      `${origin}/reset-password?error=${encodeURIComponent("Missing verification code")}`,
    );
  }

  try {
    const supabase = await getServerSupabase();

    if (!supabase) {
      return NextResponse.redirect(
        `${origin}/reset-password?error=no-auth-client`,
      );
    }

    const result = await exchangePasswordResetCode(supabase, code);

    if (!result.success) {
      // Check for expired or used link errors
      // Avoid broad checks like "invalid" which can match unrelated errors (e.g., PKCE errors)
      const errorMessage = result.error?.toLowerCase() || "";
      const isExpiredOrUsed =
        errorMessage.includes("expired") ||
        errorMessage.includes("otp_expired") ||
        errorMessage.includes("already") ||
        errorMessage.includes("used");

      const errorParam = isExpiredOrUsed
        ? "link_expired"
        : encodeURIComponent(result.error || "Password reset failed");

      return NextResponse.redirect(
        `${origin}/reset-password?error=${errorParam}`,
      );
    }

    return NextResponse.redirect(`${origin}/reset-password`);
  } catch (error) {
    console.error("Password reset callback error:", error);
    return NextResponse.redirect(
      `${origin}/reset-password?error=${encodeURIComponent("Password reset failed")}`,
    );
  }
}
