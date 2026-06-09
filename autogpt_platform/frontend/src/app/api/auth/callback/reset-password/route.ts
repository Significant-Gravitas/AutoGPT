import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const token = searchParams.get("token");
  const origin =
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL || "http://localhost:3000";

  if (!token) {
    const error = searchParams.get("error");

    if (error) {
      const normalizedError = error === "INVALID_TOKEN" ? "link_expired" : error;
      return NextResponse.redirect(
        `${origin}/reset-password?error=${encodeURIComponent(normalizedError)}`,
      );
    }

    return NextResponse.redirect(
      `${origin}/reset-password?error=${encodeURIComponent("Missing verification token")}`,
    );
  }

  return NextResponse.redirect(
    `${origin}/reset-password?token=${encodeURIComponent(token)}`,
  );
}
