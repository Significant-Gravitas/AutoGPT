import { type NextRequest } from "next/server";

import { redirect } from "next/navigation";
import { auth } from "@/lib/auth/auth";
import { headers } from "next/headers";

// Email confirmation route
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const token = searchParams.get("token") ?? searchParams.get("token_hash");
  const next = searchParams.get("next") ?? "/";

  if (token) {
    try {
      await auth.api.verifyEmail({
        query: {
          token,
          callbackURL: next,
        },
        headers: new Headers(await headers()),
      });
      redirect(next);
    } catch {
      redirect("/error");
    }
  }

  redirect("/error");
}
