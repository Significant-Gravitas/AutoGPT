import { auth } from "@/lib/auth/auth";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import { mapSessionUser } from "@/lib/auth/types";
import { APIError } from "better-auth/api";
import { NextResponse } from "next/server";

export async function GET() {
  const session = await getServerSession();

  if (!session?.user) {
    return NextResponse.json({ error: "No active session" }, { status: 400 });
  }

  return NextResponse.json({ user: mapSessionUser(session.user) });
}

export async function PUT(request: Request) {
  try {
    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
    }

    const { email: rawEmail, full_name: rawFullName } = body as {
      email?: unknown;
      full_name?: unknown;
    };

    const email = typeof rawEmail === "string" ? rawEmail.trim() : undefined;
    const fullName =
      typeof rawFullName === "string" ? rawFullName.trim() : undefined;

    if (!email && !fullName) {
      return NextResponse.json(
        { error: "Email or full_name is required" },
        { status: 400 },
      );
    }

    try {
      if (fullName) {
        await auth.api.updateUser({
          body: { name: fullName },
          headers: request.headers,
        });
      }
      if (email) {
        await auth.api.changeEmail({
          body: { newEmail: email },
          headers: request.headers,
        });
      }
    } catch (error) {
      if (error instanceof APIError) {
        return NextResponse.json(
          { error: error.body?.message || error.message },
          { status: 400 },
        );
      }
      throw error;
    }

    const session = await auth.api.getSession({
      headers: request.headers,
      query: { disableCookieCache: true },
    });

    if (!session?.user) {
      return NextResponse.json({ error: "No active session" }, { status: 400 });
    }

    return NextResponse.json({ user: mapSessionUser(session.user) });
  } catch {
    return NextResponse.json(
      { error: "Failed to update user" },
      { status: 500 },
    );
  }
}
