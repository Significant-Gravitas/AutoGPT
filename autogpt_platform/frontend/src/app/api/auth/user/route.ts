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
    const { email } = await request.json();

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    const { data, error } = await supabase.auth.updateUser({
      email,
    });

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }

    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to update user email" },
      { status: 500 },
    );
  }
}
