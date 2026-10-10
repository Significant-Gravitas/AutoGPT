import { NextResponse, type NextRequest } from "next/server";
import { getPostHogProxyTarget, POSTHOG_PROXY_PATH } from "./posthog-proxy";

export function isPostHogProxyPath(pathname: string) {
  return (
    pathname === POSTHOG_PROXY_PATH ||
    pathname.startsWith(`${POSTHOG_PROXY_PATH}/`)
  );
}

// Forwards a proxy-path request to PostHog Cloud, or returns null when it
// isn't configured. Done in middleware rather than as a next.config rewrite so
// the request is cleaned first: the browser attaches every first-party cookie
// (the httpOnly session included) to same-origin requests, and a plain rewrite
// would hand them all to PostHog.
export function proxyPostHogRequest(request: NextRequest) {
  const target = getPostHogProxyTarget({
    key: process.env.NEXT_PUBLIC_POSTHOG_KEY,
    host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  });
  if (!target) return null;

  const path = request.nextUrl.pathname.slice(POSTHOG_PROXY_PATH.length);
  const isAsset = path.startsWith("/static/") || path.startsWith("/array/");
  const destination = new URL(
    `${path}${request.nextUrl.search}`,
    isAsset ? target.assetsHost : target.apiHost,
  );

  const headers = new Headers(request.headers);
  headers.set("host", destination.host);
  headers.delete("cookie");
  headers.delete("authorization");

  return NextResponse.rewrite(destination, { request: { headers } });
}
