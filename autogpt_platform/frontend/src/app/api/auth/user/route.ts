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

    const {
      email: rawEmail,
      full_name: rawFullName,
      preferred_name: rawPreferredName,
    } = body as {
      email?: unknown;
      full_name?: unknown;
      preferred_name?: unknown;
    };

    const email = typeof rawEmail === "string" ? rawEmail.trim() : undefined;
    const fullName =
      typeof rawFullName === "string" ? rawFullName.trim() : undefined;
    const preferredName =
      typeof rawPreferredName === "string"
        ? rawPreferredName.trim()
        : undefined;

    if (!email && !fullName && !preferredName) {
      return NextResponse.json(
        { error: "Email, full_name or preferred_name is required" },
        { status: 400 },
      );
    }

    // The two writes below can't be made atomic: the profile update commits
    // immediately while the email change is a separate, verification-gated
    // flow. Combining them means a failing email change (e.g. "Email is the
    // same") reports 400 for the whole request even though the name was
    // already saved, and the client won't refetch. Reject the combination
    // instead of silently half-applying it.
    if (email && (fullName || preferredName)) {
      return NextResponse.json(
        { error: "Update email separately from profile fields" },
        { status: 400 },
      );
    }

    try {
      if (fullName || preferredName) {
        await auth.api.updateUser({
          body: {
            ...(fullName && { name: fullName }),
            ...(preferredName && { preferredName }),
          },
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
