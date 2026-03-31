import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8006";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  const { token } = await params;
  
  // Forward the authorization header (cookie or bearer token)
  const authHeader = request.headers.get("authorization");
  const cookie = request.headers.get("cookie");
  
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (authHeader) {
    headers["authorization"] = authHeader;
  }
  if (cookie) {
    headers["cookie"] = cookie;
  }
  
  const res = await fetch(`${BACKEND_URL}/api/platform-linking/tokens/${token}/confirm`, {
    method: "POST",
    headers,
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
