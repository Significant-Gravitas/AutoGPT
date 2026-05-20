import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { NextResponse } from "next/server";

export async function GET() {
  const supabase = await getServerSupabase();
  const { data, error } = await supabase.auth.getUser();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 400 });
  }

  return NextResponse.json(data);
}

export async function PUT(request: Request) {
  try {
    const supabase = await getServerSupabase();

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

    const updatePayload: Parameters<typeof supabase.auth.updateUser>[0] = {};
    if (email) updatePayload.email = email;
    if (fullName) updatePayload.data = { full_name: fullName };

    const { data, error } = await supabase.auth.updateUser(updatePayload);

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to update user" },
      { status: 500 },
    );
  }
}
