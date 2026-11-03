import type { Event } from "@sentry/nextjs";

// Browser envelopes go to Sentry through our own domain so ad blockers
// don't drop them. Misspelt on purpose: blocklist rules match the literal word.
export const SENTRY_TUNNEL_PATH = "/relay/snentry";

const SENTRY_INGEST_HOST = /^o(\d+)\.ingest(?:\.([a-z]{2}))?\.sentry\.io$/;

// Carries the DSN's org, project and region as `o`, `p` and `r`, the query
// @sentry/nextjs's own tunnel uses; middleware rebuilds the ingest URL from it.
export function getSentryTunnelURL(dsn: string | undefined) {
  if (!dsn) return undefined;
  const { hostname, pathname } = new URL(dsn);
  const [, org, region] = hostname.match(SENTRY_INGEST_HOST) ?? [];
  if (!org) return undefined;

  const query = new URLSearchParams({ o: org, p: pathname.slice(1) });
  if (region) query.set("r", region);
  return `${SENTRY_TUNNEL_PATH}?${query}`;
}

// Sentry dropped these itself while the tunnel was its `tunnelRoute`. The server
// transaction carries the request URL; the edge middleware's only its path,
// which Sentry's spans name `url.path` and Next's own `http.target`.
export function isSentryTunnelTransaction(event: Event) {
  const data = event.contexts?.trace?.data;
  return [event.request?.url, data?.["url.path"], data?.["http.target"]].some(
    (url) => typeof url === "string" && getPathname(url) === SENTRY_TUNNEL_PATH,
  );
}

// A throwing beforeSendTransaction would drop the event, so a malformed URL is no match.
function getPathname(url: string) {
  try {
    return new URL(url, "http://localhost").pathname;
  } catch {
    return null;
  }
}
