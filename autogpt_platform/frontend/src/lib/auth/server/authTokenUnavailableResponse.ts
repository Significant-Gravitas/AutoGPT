import type { AuthTokenUnavailableError } from "@/lib/autogpt-server-api/helpers";
import { NextResponse } from "next/server";

// What a route handler answers instead of forwarding a request it could not
// authenticate. The client retries a 503; a 401 would sign the user out.
export function authTokenUnavailableResponse(error: AuthTokenUnavailableError) {
  return NextResponse.json(error.response, {
    status: error.status,
    headers: { "Retry-After": "2" },
  });
}
