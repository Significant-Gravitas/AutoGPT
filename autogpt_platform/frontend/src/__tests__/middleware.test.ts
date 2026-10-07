import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const authMiddlewareMock = vi.fn();

vi.mock("@/lib/auth/middleware", () => ({
  authMiddleware: (...args: unknown[]) => authMiddlewareMock(...args),
}));

import { config, middleware } from "../middleware";

beforeEach(() => {
  authMiddlewareMock.mockReset();
});

describe("middleware www→non-www redirect", () => {
  it("redirects www host to non-www with 308", async () => {
    const request = new NextRequest("https://www.example.com/dashboard");

    const response = await middleware(request);

    expect(response).toBeDefined();
    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/dashboard",
    );
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("treats uppercase WWW host as case-insensitive (URL API normalizes)", async () => {
    const request = new NextRequest("https://WWW.example.com/path?x=1");

    const response = await middleware(request);

    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/path?x=1",
    );
  });

  it("falls through to authMiddleware when host is non-www", async () => {
    const passthrough = new Response("ok");
    authMiddlewareMock.mockResolvedValueOnce(passthrough);

    const request = new NextRequest("https://example.com/dashboard");
    const response = await middleware(request);

    expect(authMiddlewareMock).toHaveBeenCalledTimes(1);
    expect(response).toBe(passthrough);
  });
});

describe("middleware trailing-slash redirect", () => {
  it("strips a trailing slash from page paths with 308", async () => {
    const response = await middleware(
      new NextRequest("https://example.com/marketplace/?q=1"),
    );

    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/marketplace?q=1",
    );
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("fixes www and the trailing slash in one redirect", async () => {
    const response = await middleware(
      new NextRequest("https://www.example.com/copilot/"),
    );

    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/copilot",
    );
  });

  it("leaves the root path alone", async () => {
    authMiddlewareMock.mockResolvedValueOnce(new Response("ok"));

    await middleware(new NextRequest("https://example.com/"));

    expect(authMiddlewareMock).toHaveBeenCalledTimes(1);
  });
});

describe("middleware PostHog proxy", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  function proxiedRequest(url: string) {
    return new NextRequest(url, {
      method: "POST",
      headers: {
        cookie: "better-auth.session_token=secret; sb-proj-auth-token=legacy",
        authorization: "Bearer secret",
        "content-type": "application/json",
      },
    });
  }

  it("runs middleware on the proxy path", () => {
    expect(
      unstable_doesMiddlewareMatch({
        config,
        url: "https://platform.agpt.co/relay/i/v0/e/",
      }),
    ).toBe(true);
  });

  it("forwards events to PostHog, trailing slash and query intact", async () => {
    const response = await middleware(
      proxiedRequest("https://platform.agpt.co/relay/i/v0/e/?ip=0&_=1"),
    );

    expect(response.headers.get("x-middleware-rewrite")).toBe(
      "https://eu.i.posthog.com/i/v0/e/?ip=0&_=1",
    );
    expect(response.headers.get("location")).toBeNull();
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("serves PostHog's static assets from its assets host", async () => {
    const asset = await middleware(
      new NextRequest("https://platform.agpt.co/relay/static/array.js"),
    );
    const remoteConfig = await middleware(
      new NextRequest("https://platform.agpt.co/relay/array/phc_test/config"),
    );

    expect(asset.headers.get("x-middleware-rewrite")).toBe(
      "https://eu-assets.i.posthog.com/static/array.js",
    );
    expect(remoteConfig.headers.get("x-middleware-rewrite")).toBe(
      "https://eu-assets.i.posthog.com/array/phc_test/config",
    );
  });

  it("never hands our cookies or auth header to PostHog", async () => {
    const response = await middleware(
      proxiedRequest("https://platform.agpt.co/relay/flags/?v=2"),
    );

    const forwarded = (
      response.headers.get("x-middleware-override-headers") ?? ""
    ).split(",");
    expect(forwarded).toContain("content-type");
    expect(forwarded).not.toContain("cookie");
    expect(forwarded).not.toContain("authorization");
    expect(response.headers.get("x-middleware-request-host")).toBe(
      "eu.i.posthog.com",
    );
  });

  it("skips the www redirect and auth checks on the proxy path", async () => {
    const response = await middleware(
      proxiedRequest("https://www.platform.agpt.co/relay/e/"),
    );

    expect(response.status).not.toBe(308);
    expect(response.headers.get("x-middleware-rewrite")).toBe(
      "https://eu.i.posthog.com/e/",
    );
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("passes the path through untouched when PostHog Cloud isn't configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://posthog.example.com");

    const response = await middleware(
      proxiedRequest("https://platform.agpt.co/relay/e/"),
    );

    expect(response.headers.get("x-middleware-rewrite")).toBeNull();
    expect(response.headers.get("x-middleware-next")).toBe("1");
    expect(authMiddlewareMock).not.toHaveBeenCalled();
  });

  it("doesn't treat look-alike paths as the proxy", async () => {
    authMiddlewareMock.mockResolvedValueOnce(new Response("ok"));

    await middleware(new NextRequest("https://platform.agpt.co/relayed"));

    expect(authMiddlewareMock).toHaveBeenCalledTimes(1);
  });
});
