import { NextResponse, type NextRequest } from "next/server";
import { getPostHogProxyTarget, POSTHOG_PROXY_PATH } from "./posthog-proxy";

export function isPostHogProxyPath(pathname: string) {
  return (
    pathname === POSTHOG_PROXY_PATH ||
    pathname.startsWith(`${POSTHOG_PROXY_PATH}/`)
  );
}

// Forwards a proxy-path request to PostHog Cloud (null when not configured).
// Middleware, not a next.config rewrite: a rewrite forwards every header.
export function proxyPostHogRequest(request: NextRequest) {
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  const target = getPostHogProxyTarget({
    key,
    host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  });
  if (!key || !target) return null;

  const path = request.nextUrl.pathname.slice(POSTHOG_PROXY_PATH.length);
  const upstream = getUpstreamHost({ path, key, target });
  if (!upstream) return new NextResponse(null, { status: 404 });

  // Never resolve the path against the host: "//evil.com/x" would replace it.
  const destination = new URL(upstream);
  destination.pathname = path;
  destination.search = request.nextUrl.search;

  const headers = new Headers({ host: destination.host });
  for (const name of FORWARDED_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }

  return NextResponse.rewrite(destination, { request: { headers } });
}

interface UpstreamArgs {
  path: string;
  key: string;
  target: NonNullable<ReturnType<typeof getPostHogProxyTarget>>;
}

// Any other path on PostHog's hosts would be served as our own origin.
function getUpstreamHost({ path, key, target }: UpstreamArgs) {
  if (API_PATHS.has(path)) return target.apiHost;
  const isAsset =
    path.startsWith("/static/") ||
    path === `/array/${key}/config` ||
    path === `/array/${key}/config.js`;
  return isAsset ? target.assetsHost : null;
}

// What posthog-js calls for capture, session replay, feature flags and surveys.
const API_PATHS = new Set([
  "/e/",
  "/i/v0/e/",
  "/s/",
  "/flags/",
  "/api/surveys/",
]);

// An allowlist: no cookie or auth header, now or added later, reaches PostHog.
const FORWARDED_HEADERS = [
  "accept",
  "accept-encoding",
  "content-encoding",
  "content-length",
  "content-type",
  "user-agent",
  // PostHog places the visitor by IP.
  "x-forwarded-for",
];
