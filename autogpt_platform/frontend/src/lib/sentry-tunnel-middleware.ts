import { getSentryTunnelURL } from "@/lib/sentry-tunnel";
import { environment } from "@/services/environment";
import { NextResponse, type NextRequest } from "next/server";

// What Sentry's ingest reads. The forwarded-for headers keep the user's IP,
// which Sentry records; cookies and Authorization never leave.
const FORWARDED_HEADERS = [
  "content-type",
  "content-length",
  "content-encoding",
  "user-agent",
  "sec-ch-ua",
  "sec-ch-ua-model",
  "sec-ch-ua-platform",
  "sec-ch-ua-platform-version",
  "origin",
  "referer",
  "x-forwarded-for",
  "x-vercel-forwarded-for",
];

// Above Replay's ~20 MB event buffer, the largest envelope the browser SDK sends.
const MAX_ENVELOPE_BYTES = 20 * 1024 * 1024;

export function relaySentryEnvelope(request: NextRequest) {
  const destination = getIngestURL(request);
  if (!destination) return new NextResponse(null, { status: 404 });
  if (request.method !== "POST") {
    return new NextResponse(null, { status: 405, headers: { allow: "POST" } });
  }
  const length = Number(request.headers.get("content-length"));
  if (!length || length > MAX_ENVELOPE_BYTES) {
    return new NextResponse(null, { status: 413 });
  }

  const headers = new Headers({ host: destination.host });
  for (const name of FORWARDED_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }

  return NextResponse.rewrite(destination, { request: { headers } });
}

// Only our own DSN's tunnel URL is relayed, so the path carries nothing to
// other Sentry orgs or projects.
function getIngestURL(request: NextRequest) {
  const { pathname, search, searchParams: query } = request.nextUrl;
  if (pathname + search !== getSentryTunnelURL(environment.getSentryDSN())) {
    return null;
  }

  const org = query.get("o") ?? "";
  const project = query.get("p") ?? "";
  const region = query.get("r");
  if (!/^\d+$/.test(org) || !/^\d+$/.test(project)) return null;
  if (region !== null && !/^[a-z]{2}$/.test(region)) return null;

  const host = region
    ? `o${org}.ingest.${region}.sentry.io`
    : `o${org}.ingest.sentry.io`;
  return new URL(`https://${host}/api/${project}/envelope/?hsts=0`);
}
