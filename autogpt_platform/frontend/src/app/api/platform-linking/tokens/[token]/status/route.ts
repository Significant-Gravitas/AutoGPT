import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8006";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  const { token } = await params;
  
  const res = await fetch(`${BACKEND_URL}/api/platform-linking/tokens/${token}/status`, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
