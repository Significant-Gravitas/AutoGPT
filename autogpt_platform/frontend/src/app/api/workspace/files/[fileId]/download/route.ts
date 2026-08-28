import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { environment } from "@/services/environment";
import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ fileId: string }> },
) {
  const { fileId } = await params;
  const token = await getServerAuthToken();
  const backendUrl = new URL(
    `/api/workspace/files/${encodeURIComponent(fileId)}/download`,
    environment.getAGPTServerBaseUrl(),
  );
  const headers = new Headers({
    Accept: request.headers.get("accept") ?? "*/*",
  });
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const organizationId = request.nextUrl.searchParams.get("organizationId");
  const teamId = request.nextUrl.searchParams.get("teamId");
  if (organizationId) headers.set(ORG_HEADER_NAME, organizationId);
  if (teamId !== null) headers.set(TEAM_HEADER_NAME, teamId);

  const response = await fetch(backendUrl, { headers });
  const body = await response.arrayBuffer();
  const responseHeaders = new Headers();
  const contentType = response.headers.get("content-type");
  const contentDisposition = response.headers.get("content-disposition");
  if (contentType) responseHeaders.set("content-type", contentType);
  if (contentDisposition) {
    responseHeaders.set("content-disposition", contentDisposition);
  }
  responseHeaders.set("content-length", String(body.byteLength));

  return new NextResponse(body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}
