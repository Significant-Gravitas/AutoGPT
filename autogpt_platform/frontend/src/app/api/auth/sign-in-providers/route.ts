import { auth } from "@/lib/auth/auth";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import { NextResponse } from "next/server";

/**
 * The providers this account signs in with, by name only. Better Auth's own
 * `/list-accounts` stays disabled (disabled-paths.ts) because it hands the
 * browser account details; onboarding only needs to know which one it was.
 */
export async function GET(request: Request) {
  const session = await getServerSession();
  if (!session?.user) {
    return NextResponse.json({ error: "No active session" }, { status: 401 });
  }

  const accounts = await auth.api.listUserAccounts({
    headers: request.headers,
  });
  const providers = Array.from(
    new Set(accounts.map((account) => account.providerId)),
  );
  return NextResponse.json({ providers });
}
