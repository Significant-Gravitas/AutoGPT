import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const next = searchParams.get("next") ?? "/dashboard";
  const api = new AutoGPTServerAPI();

  if (code && state) {
    try {
      //TODO kcze use proper provider
      await api.oAuthCallback("github", code, state);
      //TODO kcze use proper next
      return NextResponse.redirect(`${origin}${next}`);
    } catch (error) {
      console.error("OAuth callback error:", error);
      return NextResponse.redirect(`${origin}/auth/login-error`);
    }
  }

  return NextResponse.redirect(`${origin}/auth/auth-code-error`);
}
