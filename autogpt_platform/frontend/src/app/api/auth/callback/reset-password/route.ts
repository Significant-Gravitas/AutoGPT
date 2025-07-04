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
      `${origin}/reset-password?error=Missing verification code`,
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
      return NextResponse.redirect(
        `${origin}/reset-password?error=${encodeURIComponent(result.error || "Password reset failed")}`,
      );
    }

    return NextResponse.redirect(`${origin}/reset-password`);
  } catch (error) {
    console.error("Password reset callback error:", error);
    return NextResponse.redirect(
      `${origin}/reset-password?error=Password reset failed`,
    );
  }
}
