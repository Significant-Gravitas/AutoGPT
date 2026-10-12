// @vitest-environment node
// happy-dom hides a Request's cookie header from get(), so a cookie test there cannot fail.
import { getSentryTunnelURL, SENTRY_TUNNEL_PATH } from "@/lib/sentry-tunnel";
import { readFileSync } from "node:fs";
import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const authMiddlewareMock = vi.fn();

vi.mock("@/lib/auth/middleware", () => ({
  authMiddleware: (...args: unknown[]) => authMiddlewareMock(...args),
}));

import { config, middleware } from "../middleware";

const ORIGIN = "https://platform.agpt.co";
const DSN =
  "https://key@o4505260022104064.ingest.us.sentry.io/4507946746380288";

beforeEach(() => {
  authMiddlewareMock.mockReset();
  vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", DSN);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Sentry tunnel", () => {
  it("runs through our middleware, not Sentry's own tunnel rewrite", () => {
    const nextConfig = readFileSync(
      new URL("../../next.config.mjs", import.meta.url),
      "utf8",
    );

    expect(nextConfig).not.toMatch(/^\s*tunnelRoute:/m);
    expect(
      unstable_doesMiddlewareMatch({
        config,
        url: `${ORIGIN}${getSentryTunnelURL(DSN)}`,
      }),
    ).toBe(true);
  });

  it.each([
    [
      DSN,
      "https://o4505260022104064.ingest.us.sentry.io/api/4507946746380288/envelope/?hsts=0",
    ],
    [
      "https://key@o1.ingest.sentry.io/2",
      "https://o1.ingest.sentry.io/api/2/envelope/?hsts=0",
    ],
  ])("forwards %s's envelopes to its ingest", async (dsn, destination) => {
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", dsn);
    const response = await middleware(envelopePost(getSentryTunnelURL(dsn)));

    expect(response.headers.get("x-middleware-rewrite")).toBe(destination);
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("forwards only the headers Sentry reads", async () => {
    const response = await middleware(envelopePost(getSentryTunnelURL(DSN)));

    expect(forwardedHeaders(response)).toEqual({
      host: "o4505260022104064.ingest.us.sentry.io",
      "content-type": "text/plain;charset=UTF-8",
      "content-length": "3",
      "user-agent": "Mozilla/5.0",
      "x-forwarded-for": "203.0.113.7",
    });
  });

  it.each([
    "o=1&p=2",
    "o=1&p=2&r=evil.example",
    "o=evil.example&p=2",
    "o=1",
    "p=2",
  ])("refuses ?%s without forwarding it", async (query) => {
    const response = await middleware(
      envelopePost(`${SENTRY_TUNNEL_PATH}?${query}`),
    );

    expect(response.status).toBe(404);
    expect(response.headers.get("x-middleware-rewrite")).toBeNull();
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("relays nothing without a configured DSN", async () => {
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "");
    const response = await middleware(envelopePost(getSentryTunnelURL(DSN)));

    expect(response.status).toBe(404);
    expect(response.headers.get("x-middleware-rewrite")).toBeNull();
  });

  it("relays POST only", async () => {
    const response = await middleware(
      new NextRequest(`${ORIGIN}${getSentryTunnelURL(DSN)}`, { method: "GET" }),
    );

    expect(response.status).toBe(405);
    expect(response.headers.get("allow")).toBe("POST");
    expect(response.headers.get("x-middleware-rewrite")).toBeNull();
  });

  it.each([null, "0", "abc", String(21 * 1024 * 1024)])(
    "refuses a content-length of %s",
    async (contentLength) => {
      const response = await middleware(
        envelopePost(getSentryTunnelURL(DSN), contentLength),
      );

      expect(response.status).toBe(413);
      expect(response.headers.get("x-middleware-rewrite")).toBeNull();
    },
  );

  it("leaves a self-hosted DSN untunnelled", () => {
    expect(getSentryTunnelURL("https://key@sentry.example.com/2")).toBe(
      undefined,
    );
  });
});

function envelopePost(
  tunnelURL: string | undefined,
  contentLength: string | null = "3",
) {
  return new NextRequest(`${ORIGIN}${tunnelURL}`, {
    method: "POST",
    body: "{}\n",
    headers: {
      cookie: "better-auth.session_token=secret; sb-proj-auth-token=legacy",
      authorization: "Bearer secret",
      "content-type": "text/plain;charset=UTF-8",
      "user-agent": "Mozilla/5.0",
      "x-forwarded-for": "203.0.113.7",
      "x-unlisted": "secret",
      ...(contentLength === null ? {} : { "content-length": contentLength }),
    },
  });
}

function forwardedHeaders(response: Response) {
  const names = (response.headers.get("x-middleware-override-headers") ?? "")
    .split(",")
    .filter(Boolean);
  return Object.fromEntries(
    names.map((name) => [
      name,
      response.headers.get(`x-middleware-request-${name}`),
    ]),
  );
}
