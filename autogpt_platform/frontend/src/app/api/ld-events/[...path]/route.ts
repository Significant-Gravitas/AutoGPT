import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const LAUNCHDARKLY_EVENTS_ORIGIN = "https://events.launchdarkly.com";

// `events.launchdarkly.com` is on every tracker blocklist, so a flush straight
// to it is rejected in the browser and printed twice to the console. The LD
// provider's `eventsUrl` points here instead.
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const path = (await params).path.join("/");
  if (!isEventsPath(path)) {
    return new NextResponse(null, { status: 404 });
  }

  try {
    const upstream = await fetch(`${LAUNCHDARKLY_EVENTS_ORIGIN}/${path}`, {
      method: "POST",
      headers: forwardedHeaders(request),
      body: await request.text(),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    return new NextResponse(null, { status: upstream.status });
  } catch {
    // Analytics are fire-and-forget: a dropped batch must never surface to the
    // user, and the SDK would retry a 5xx once for nothing.
    return new NextResponse(null, { status: 202 });
  }
}

const UPSTREAM_TIMEOUT_MS = 10_000;

// The SDK only ever posts these two. Anything else would make this an open
// proxy onto LaunchDarkly's event API.
function isEventsPath(path: string): boolean {
  return /^events\/(bulk|diagnostic)\/[A-Za-z0-9]+$/.test(path);
}

// Same-origin requests carry our session cookies; LaunchDarkly has no business
// receiving them, so only what the event API reads is forwarded.
function forwardedHeaders(request: NextRequest): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  for (const name of [
    "x-launchdarkly-event-schema",
    "x-launchdarkly-payload-id",
  ]) {
    const value = request.headers.get(name);
    if (value) headers[name] = value;
  }
  return headers;
}
