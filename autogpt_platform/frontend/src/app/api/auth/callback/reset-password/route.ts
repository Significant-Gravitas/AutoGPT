import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const token = searchParams.get("token");
  const origin =
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";

  if (!token) {
    return NextResponse.redirect(
      `${origin}/reset-password?error=Missing verification token`,
    );
  }

  try {
    // Verify the token is valid with the backend
    const response = await fetch(
      `${API_BASE_URL}/api/auth/password-reset/verify?token=${encodeURIComponent(token)}`,
      {
        method: "GET",
      },
    );

    if (!response.ok) {
      const data = await response.json();
      return NextResponse.redirect(
        `${origin}/reset-password?error=${encodeURIComponent(data.detail || "Invalid or expired reset link")}`,
      );
    }

    // Store the reset token in a cookie for the password change form
    const cookieStore = await cookies();
    cookieStore.set("password_reset_token", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 15, // 15 minutes
      path: "/",
    });

    return NextResponse.redirect(`${origin}/reset-password`);
  } catch (error) {
    console.error("Password reset callback error:", error);
    return NextResponse.redirect(
      `${origin}/reset-password?error=Password reset failed`,
    );
  }
}
