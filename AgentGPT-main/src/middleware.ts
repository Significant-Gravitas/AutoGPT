import type { NextRequest } from "next/server";
import { ipAddress } from "@vercel/edge";
import { rateLimiter } from "./server/redis";

export const config = {
  // Only run the middleware on the home route
  matcher: "/api/agent/:path*",
};

function ipFallback(request: Request) {
  const xff = request.headers.get("x-forwarded-for");
  return xff ? (Array.isArray(xff) ? xff[0] : xff.split(",")[0]) : "127.0.0.1";
}

async function shouldRateLimit(request: NextRequest): Promise<boolean> {
  const ip = ipAddress(request) || ipFallback(request) || "";
  if (!ip) {
    return false;
  }

  const { success } = await rateLimiter.limit(ip);
  return !success;
}

const rateLimitedResponse = () =>
  new Response("Too many requests, please try again later.", {
    status: 429,
  });

// noinspection JSUnusedGlobalSymbols
export async function middleware(request: NextRequest) {
  if (await shouldRateLimit(request)) {
    return rateLimitedResponse();
  }
}
