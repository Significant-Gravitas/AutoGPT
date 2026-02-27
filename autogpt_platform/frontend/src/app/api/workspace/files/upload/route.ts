import { environment } from "@/services/environment";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const sessionId = request.nextUrl.searchParams.get("session_id");

    const token = await getServerAuthToken();
    const backendUrl = environment.getAGPTServerBaseUrl();

    const uploadUrl = new URL("/api/workspace/files/upload", backendUrl);
    if (sessionId) {
      uploadUrl.searchParams.set("session_id", sessionId);
    }

    const headers: Record<string, string> = {};
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(uploadUrl.toString(), {
      method: "POST",
      headers,
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      return new NextResponse(errorText, {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("File upload proxy error:", error);
    return NextResponse.json(
      {
        error: "Failed to upload file",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}
