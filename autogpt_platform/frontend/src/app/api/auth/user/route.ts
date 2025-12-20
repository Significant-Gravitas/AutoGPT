import {
  getServerAuthToken,
  getServerUser,
} from "@/lib/auth/server/getServerAuth";
import { environment } from "@/services/environment";
import { NextResponse } from "next/server";

export async function GET() {
  const user = await getServerUser();

  if (!user) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  return NextResponse.json({ user });
}

export async function PUT(request: Request) {
  try {
    const token = await getServerAuthToken();

    if (!token) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const { email } = await request.json();

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/update-email`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ email }),
      },
    );

    if (!response.ok) {
      const data = await response.json();
      return NextResponse.json(
        { error: data.detail || "Failed to update email" },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Failed to update user email" },
      { status: 500 },
    );
  }
}
