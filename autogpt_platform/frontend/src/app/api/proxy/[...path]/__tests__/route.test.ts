// @vitest-environment node
// Node's own fetch, as the route gets in production: the dropped-connection
// tests send it to a real server.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

vi.mock("@/lib/auth/server/getServerAuthToken", () => ({
  getServerAuthToken: vi.fn(),
}));

vi.mock("@/lib/auth/country-token", () => ({
  getCountryToken: vi.fn(),
}));

vi.mock("@/lib/auth/client-ip-token", () => ({
  getClientIPToken: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: vi.fn(() => "https://backend.test"),
    isClientSide: vi.fn(() => false),
    isServerSide: vi.fn(() => true),
  },
}));

vi.mock("@/lib/auth/server/getServerSession", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/lib/auth/server/getServerSession")
  >()),
  readServerSession: vi.fn(),
}));

import { getClientIPToken } from "@/lib/auth/client-ip-token";
import { getCountryToken } from "@/lib/auth/country-token";
import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { readServerSession } from "@/lib/auth/server/getServerSession";
import { AuthTokenUnavailableError } from "@/lib/autogpt-server-api/helpers";
import { APIError } from "better-auth/api";
import { encodeImpersonation } from "@/lib/impersonation";
import { environment } from "@/services/environment";
import {
  hangUp,
  startLoopbackBackend,
} from "@/tests/integrations/loopback-backend";
import { DELETE, GET, PATCH, POST, PUT } from "../route";

const PRIVATE_IMAGE_HEADER_KEYS = [
  "accept",
  "accept-encoding",
  "authorization",
  "cache-control",
];

const BACKEND = "https://backend.test";

function makeParams(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

describe("proxy route — handler pass-through", () => {
  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockResolvedValue("test-token");
    vi.mocked(getCountryToken).mockReset();
    vi.mocked(getCountryToken).mockImplementation(
      async (country) => `signed-${country}`,
    );
    vi.mocked(getClientIPToken).mockReset();
    vi.mocked(getClientIPToken).mockImplementation(
      async (headers) => `signed-ip-${headers.get("x-forwarded-for")}`,
    );
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("forwards GET with Authorization header injected from server token", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ hello: "world" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(fetch).toHaveBeenCalledOnce();
    const [calledUrl, init] = vi.mocked(fetch).mock.calls[0];
    expect(calledUrl).toBe(`${BACKEND}/api/v1/items`);
    const sentHeaders = init!.headers as Headers;
    expect(sentHeaders.get("authorization")).toBe("Bearer test-token");

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ hello: "world" });
  });

  it("preserves status, statusText, and body on 4xx error responses", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ detail: "not found" }), {
        status: 404,
        statusText: "Not Found",
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/missing");
    const res = await GET(req, makeParams(["api", "v1", "missing"]));

    expect(res.status).toBe(404);
    expect(res.statusText).toBe("Not Found");
    expect(await res.json()).toEqual({ detail: "not found" });
  });

  it("passes 204 No Content through with empty body even when backend sets Content-Type: application/json", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(null, {
        status: 204,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/x", {
      method: "DELETE",
    });
    const res = await DELETE(req, makeParams(["api", "v1", "x"]));

    expect(res.status).toBe(204);
    const text = await res.text();
    expect(text).toBe("");
  });

  it("strips content-length and content-encoding from response headers but preserves others", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": "1234",
          "Content-Encoding": "gzip",
          Authorization: "Bearer upstream-secret",
          "Cache-Control": "no-store",
          "X-Custom-Header": "preserved",
        },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.headers.get("content-encoding")).toBeNull();
    expect(res.headers.get("content-length")).toBeNull();
    expect(res.headers.get("authorization")).toBeNull();
    expect(res.headers.get("cache-control")).toBe("no-store");
    expect(res.headers.get("x-custom-header")).toBe("preserved");
    expect(res.headers.get("content-type")).toBe("application/json");
  });

  it("strips Set-Cookie so backend cookies cannot attach to the frontend origin", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Set-Cookie": "session=leak; Path=/; HttpOnly",
        },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.headers.get("set-cookie")).toBeNull();
  });

  it("forwards POST body with duplex: 'half' for streaming", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ created: true }), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      method: "POST",
      body: JSON.stringify({ name: "foo" }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await POST(req, makeParams(["api", "v1", "items"]));

    const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit & {
      duplex?: string;
    };
    expect(init.method).toBe("POST");
    expect(init.duplex).toBe("half");
    expect(init.body).toBeDefined();
    expect(res.status).toBe(201);
  });

  it("does NOT forward the request Cookie header to the backend", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: {
        Cookie: "sb-access-token=secret; sb-refresh-token=secret-refresh",
      },
    });
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("cookie")).toBeNull();
    expect(sentHeaders.get("authorization")).toBe("Bearer test-token");
  });

  it("forwards allowlisted Sentry trace headers", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: {
        "sentry-trace": "abc123-def456-1",
        baggage: "sentry-trace_id=abc123",
      },
    });
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("sentry-trace")).toBe("abc123-def456-1");
    expect(sentHeaders.get("baggage")).toBe("sentry-trace_id=abc123");
  });

  it("forwards allowlisted DataFast and PostHog session headers", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: {
        "x-datafast-visitor-id": "visitor-123",
        "x-datafast-session-id": "session-456",
        "x-posthog-session-id": "posthog-session-789",
      },
    });
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("x-datafast-visitor-id")).toBe("visitor-123");
    expect(sentHeaders.get("x-datafast-session-id")).toBe("session-456");
    expect(sentHeaders.get("x-posthog-session-id")).toBe("posthog-session-789");
  });

  it("sends the edge-observed country only as a signed token", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: { "x-vercel-ip-country": "DE" },
    });
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(getCountryToken).toHaveBeenCalledWith("DE");
    expect(sentHeaders.get("x-client-country-token")).toBe("signed-DE");
    expect(sentHeaders.get("x-client-country")).toBeNull();
    expect(sentHeaders.get("x-vercel-ip-country")).toBeNull();
  });

  it("drops a browser-supplied country, plain or tokenised", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: {
        "x-client-country": "US",
        "x-client-country-token": "forged",
      },
    });
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("x-client-country")).toBeNull();
    expect(sentHeaders.get("x-client-country-token")).toBeNull();
    expect(getCountryToken).not.toHaveBeenCalled();
  });

  it("still proxies, without a country, when signing fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(getCountryToken).mockRejectedValueOnce(new Error("no jwks"));
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: { "x-vercel-ip-country": "DE" },
    });
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.status).toBe(200);
    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("x-client-country-token")).toBeNull();
  });

  it.each([
    ["api/store/search", "?query=sheets"],
    ["api/store/agents", "?search_query=sheets"],
  ])(
    "sends the visitor's IP to %s only as the proxy's own signed token",
    async (route, query) => {
      vi.mocked(fetch).mockResolvedValue(
        new Response("{}", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );

      const req = new NextRequest(
        `https://app.test/api/proxy/${route}${query}`,
        {
          headers: {
            "x-forwarded-for": "203.0.113.7",
            "x-client-ip-token": "forged",
          },
        },
      );
      await GET(req, makeParams(route.split("/")));

      const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
      expect(sentHeaders.get("x-client-ip-token")).toBe(
        "signed-ip-203.0.113.7",
      );
      expect(sentHeaders.get("x-forwarded-for")).toBeNull();
    },
  );

  it.each([
    ["api/store/agents", ""],
    ["api/store/agents", "?search_query=%20"],
    ["api/v1/items", "?query=sheets"],
  ])(
    "sends no client IP token to %s%s, even one the browser sent",
    async (route, query) => {
      vi.mocked(fetch).mockResolvedValue(
        new Response("{}", {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );

      const req = new NextRequest(
        `https://app.test/api/proxy/${route}${query}`,
        {
          headers: {
            "x-forwarded-for": "203.0.113.7",
            "x-client-ip-token": "forged",
          },
        },
      );
      await GET(req, makeParams(route.split("/")));

      const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
      expect(sentHeaders.get("x-client-ip-token")).toBeNull();
      expect(getClientIPToken).not.toHaveBeenCalled();
    },
  );

  it("omits Authorization header when no token is available", async () => {
    vi.mocked(getServerAuthToken).mockResolvedValueOnce(null);
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect(sentHeaders.get("authorization")).toBeNull();
  });

  it("answers a retryable 503 instead of forwarding a session it could not mint a token for", async () => {
    // Forwarded without a token, the request got a 401 from the backend, which
    // the browser reads as a logout (W51 N2: the auth database had run out of
    // connections for the frontend role).
    vi.mocked(getServerAuthToken).mockRejectedValueOnce(
      new AuthTokenUnavailableError(),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(fetch).not.toHaveBeenCalled();
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("2");
    expect(await res.json()).toMatchObject({ code: "auth_token_unavailable" });
  });

  it("returns 502 with error detail when backend fetch throws (network error)", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("ECONNRESET"));
    const consoleErr = vi.spyOn(console, "error").mockImplementation(() => {});

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.status).toBe(502);
    const body = await res.json();
    expect(body).toEqual({
      error: "Proxy request failed",
      detail: "ECONNRESET",
    });
    consoleErr.mockRestore();
  });

  it("sends accept-encoding restricted to encodings undici (Node 22) can decompress — must NOT include zstd", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    await GET(req, makeParams(["api", "v1", "items"]));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    const acceptEncoding = sentHeaders.get("accept-encoding") ?? "";
    expect(acceptEncoding).not.toBe("");
    expect(acceptEncoding.toLowerCase()).not.toContain("zstd");
    expect(acceptEncoding.toLowerCase()).toMatch(/gzip|br|deflate/);
  });

  it("forwards the Expert appearance purpose on uploads", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response('"https://cdn.test/avatar.png"', { status: 200 }),
    );
    const body = new FormData();
    body.append(
      "file",
      new Blob(["image"], { type: "image/png" }),
      "avatar.png",
    );
    const req = new NextRequest(
      "https://app.test/api/proxy/api/store/submissions/media?purpose=expert-avatar",
      { method: "POST", body },
    );
    await POST(req, makeParams(["api", "store", "submissions", "media"]));
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
      `${BACKEND}/api/store/submissions/media?purpose=expert-avatar`,
    );
    expect(vi.mocked(fetch).mock.calls[0][1]!.method).toBe("POST");
  });

  it("forwards query string to the backend URL", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest(
      "https://app.test/api/proxy/api/v1/items?page=2&size=20",
    );
    await GET(req, makeParams(["api", "v1", "items"]));

    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
      `${BACKEND}/api/v1/items?page=2&size=20`,
    );
  });

  it("keeps an encoded ?, # or / inside its path segment", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest(
      "https://app.test/api/proxy/api/v1/items/a%3Fb%3Dc%23d%2Fe?page=2",
    );
    // Next.js hands the route each segment already decoded
    await GET(req, makeParams(["api", "v1", "items", "a?b=c#d/e"]));

    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
      `${BACKEND}/api/v1/items/a%3Fb%3Dc%23d%2Fe?page=2`,
    );
  });

  it("preserves private video range responses without allowing shared caching", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("video bytes", {
        status: 206,
        headers: {
          "Content-Type": "video/mp4",
          "Content-Length": "11",
          "Content-Range": "bytes 0-10/100",
          "Accept-Ranges": "bytes",
          "Cache-Control": "public, max-age=3600",
        },
      }),
    );
    const path = [
      "api",
      "store",
      "submissions",
      "media",
      "user-123",
      "videos",
      "preview.mp4",
    ];
    const req = new NextRequest(
      `https://app.test/api/proxy/${path.join("/")}`,
      {
        headers: { Range: "bytes=0-10" },
      },
    );
    const res = await GET(req, makeParams(path));

    const init = vi.mocked(fetch).mock.calls[0][1]!;
    const sentHeaders = init.headers as Headers;
    expect(sentHeaders.get("range")).toBe("bytes=0-10");
    expect(sentHeaders.get("accept-encoding")).toBe("identity");
    expect(init.cache).toBe("no-store");
    expect(res.status).toBe(206);
    expect(await res.text()).toBe("video bytes");
    expect(res.headers.get("content-length")).toBe("11");
    expect(res.headers.get("content-range")).toBe("bytes 0-10/100");
    expect(res.headers.get("accept-ranges")).toBe("bytes");
    expect(res.headers.get("cache-control")).toBe("private, no-store");
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("forwards only allowlisted headers for a private image", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("image", {
        status: 200,
        headers: { "Content-Type": "image/png", "Content-Length": "5" },
      }),
    );
    const path = [
      "api",
      "store",
      "submissions",
      "media",
      "user-123",
      "images",
      "thumbnail.png",
    ];
    const req = new NextRequest(
      `https://app.test/api/proxy/${path.join("/")}`,
      {
        headers: {
          Accept: "image/*",
          Authorization: "Bearer client-supplied",
          Range: "bytes=0-10",
          "X-Forwarded-For": "203.0.113.7",
        },
      },
    );
    await GET(req, makeParams(path));

    const sentHeaders = vi.mocked(fetch).mock.calls[0][1]!.headers as Headers;
    expect([...sentHeaders.keys()].sort()).toEqual(PRIVATE_IMAGE_HEADER_KEYS);
    expect(sentHeaders.get("authorization")).not.toBe("Bearer client-supplied");
  });

  it("forwards the backend signed redirect for oversized private media", async () => {
    const signedUrl =
      "https://storage.googleapis.com/private/object?signature=test";
    vi.mocked(fetch).mockResolvedValue(
      new Response(null, {
        status: 307,
        headers: {
          Location: signedUrl,
          "Cache-Control": "private, no-store",
        },
      }),
    );
    const path = [
      "api",
      "store",
      "submissions",
      "media",
      "user-123",
      "images",
      "large.png",
    ];
    const res = await GET(
      new NextRequest(`https://app.test/api/proxy/${path.join("/")}`),
      makeParams(path),
    );

    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toBe(signedUrl);
    expect(res.headers.get("cache-control")).toBe("private, no-store");
    expect(await res.text()).toBe("");
  });

  it("does not expose private-media proxy exception details", async () => {
    vi.mocked(fetch).mockRejectedValue(
      new Error("upstream failed with bearer super-secret"),
    );
    vi.spyOn(console, "error").mockImplementation(() => {});
    const path = [
      "api",
      "store",
      "submissions",
      "media",
      "user-123",
      "images",
      "thumbnail.png",
    ];
    const req = new NextRequest(`https://app.test/api/proxy/${path.join("/")}`);
    const res = await GET(req, makeParams(path));

    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "Proxy request failed" });
    expect(res.headers.get("cache-control")).toBe("private, no-store");
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("hardens a public shared-file response after proxying storage", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("<script>window.name = 'executed'</script>", {
        status: 200,
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Content-Disposition": 'inline; filename="payload.html"',
        },
      }),
    );

    const shareToken = "550e8400-e29b-41d4-a716-446655440000";
    const fileID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
    const path = [
      "api",
      "public",
      "shared",
      shareToken,
      "files",
      fileID,
      "download",
    ];
    const req = new NextRequest(`https://app.test/api/proxy/${path.join("/")}`);
    const res = await GET(req, makeParams(path));

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("text/html; charset=utf-8");
    expect(res.headers.get("content-disposition")).toBe(
      'attachment; filename="payload.html"',
    );
    expect(res.headers.get("content-security-policy")).toBe("sandbox");
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it.each([
    "text/html; charset=utf-8",
    "image/svg+xml",
    "application/xhtml+xml",
    "text/xml",
  ])(
    "relays a %s response from any other route as a download",
    async (type) => {
      vi.mocked(fetch).mockResolvedValue(
        new Response("hello", {
          status: 200,
          headers: {
            "Content-Type": type,
            "Content-Disposition": 'inline; filename="page"',
          },
        }),
      );

      const path = ["api", "v1", "items"];
      const req = new NextRequest(
        `https://app.test/api/proxy/${path.join("/")}`,
      );
      const res = await GET(req, makeParams(path));

      expect(res.status).toBe(200);
      expect(res.headers.get("content-disposition")).toBe(
        'attachment; filename="page"',
      );
      expect(res.headers.get("x-content-type-options")).toBe("nosniff");
      expect(await res.text()).toBe("hello");
    },
  );

  it("leaves a JSON response renderable", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.headers.get("content-disposition")).toBeNull();
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("still renders the Codex device-login page", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("<p>sign in</p>", {
        status: 200,
        headers: { "Content-Type": "text/html; charset=utf-8" },
      }),
    );

    const path = ["api", "integrations", "codex", "device-login", "login-1"];
    const req = new NextRequest(`https://app.test/api/proxy/${path.join("/")}`);
    const res = await GET(req, makeParams(path));

    expect(res.headers.get("content-type")).toBe("text/html; charset=utf-8");
    expect(res.headers.get("content-disposition")).toBeNull();
  });
});

describe("proxy route — response-start timeout", () => {
  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockResolvedValue("test-token");
    vi.stubGlobal("fetch", vi.fn());
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("returns 504 when the backend never starts responding to a bodyless request", async () => {
    vi.mocked(fetch).mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init!.signal!.addEventListener("abort", () =>
            reject(init!.signal!.reason),
          );
        }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/slow");
    const resPromise = GET(req, makeParams(["api", "v1", "slow"]));

    await vi.advanceTimersByTimeAsync(31_000);

    const res = await resPromise;
    expect(res.status).toBe(504);
    expect(await res.json()).toMatchObject({
      error: "Proxy request timed out",
    });
  });
});

describe("proxy: admin impersonation header", () => {
  const ACT_AS = "x-act-as-user-id";

  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockResolvedValue("test-token");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 200 })),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function sessionAs(id: string, role: string) {
    vi.mocked(readServerSession).mockResolvedValue({
      user: { id, role },
    } as never);
  }

  async function forwardedActAs(
    cookieRecord: string | null,
    requested = "target-a",
  ) {
    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      headers: { [ACT_AS]: requested },
    });
    // The test Request drops a Cookie header, so set it on the cookie jar.
    if (cookieRecord)
      req.cookies.set("admin-impersonate-user-id", cookieRecord);
    await GET(req, makeParams(["api", "v1", "items"]));
    const [, init] = vi.mocked(fetch).mock.calls[0];
    return new Headers((init as RequestInit).headers).get(ACT_AS);
  }

  it("forwards it for the signed-in admin who chose that target", async () => {
    sessionAs("admin-1", "admin");
    expect(
      await forwardedActAs(
        encodeImpersonation({ target: "target-a", owner: "admin-1" }),
      ),
    ).toBe("target-a");
  });

  it("drops it when the session is a different admin", async () => {
    sessionAs("admin-2", "admin");
    expect(
      await forwardedActAs(
        encodeImpersonation({ target: "target-a", owner: "admin-1" }),
      ),
    ).toBeNull();
  });

  it("drops a header no impersonation cookie backs", async () => {
    sessionAs("admin-1", "admin");
    expect(await forwardedActAs(null)).toBeNull();
  });

  it("drops the target when the auth backend rejects the session", async () => {
    vi.mocked(readServerSession).mockRejectedValue(
      new APIError("UNAUTHORIZED"),
    );
    expect(
      await forwardedActAs(
        encodeImpersonation({ target: "target-a", owner: "admin-1" }),
      ),
    ).toBeNull();
  });

  it("fails the request rather than send it as the admin when the session can't be read", async () => {
    vi.mocked(readServerSession).mockRejectedValue(
      new APIError("INTERNAL_SERVER_ERROR"),
    );
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
      method: "POST",
      headers: { [ACT_AS]: "target-a" },
    });
    req.cookies.set(
      "admin-impersonate-user-id",
      encodeImpersonation({ target: "target-a", owner: "admin-1" }),
    );

    const response = await POST(req, makeParams(["api", "v1", "items"]));

    expect(response.status).toBe(502);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("drops a target the owner's cookie doesn't name", async () => {
    sessionAs("admin-1", "admin");
    expect(
      await forwardedActAs(
        encodeImpersonation({ target: "target-a", owner: "admin-1" }),
        "target-b",
      ),
    ).toBeNull();
  });
});

// A pooled connection to the backend can be dead by the time it's reused,
// and Node's fetch fails the request rather than send it again.
describe("proxy route — a dropped backend connection", () => {
  const HANDLERS = { DELETE, PATCH, POST, PUT };

  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockResolvedValue("test-token");
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "info").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.mocked(environment.getAGPTServerBaseUrl).mockReturnValue(BACKEND);
  });

  async function startBackend(
    answer: Parameters<typeof startLoopbackBackend>[0],
  ) {
    const backend = await startLoopbackBackend(answer);
    vi.mocked(environment.getAGPTServerBaseUrl).mockReturnValue(backend.url);
    return backend;
  }

  it.each([
    { method: "GET", how: "destroy", code: "UND_ERR_SOCKET" },
    { method: "GET", how: "resetAndDestroy", code: "ECONNRESET" },
    { method: "HEAD", how: "destroy", code: "UND_ERR_SOCKET" },
  ] as const)(
    "sends a $method once more when its connection is dropped ($code)",
    async ({ method, how, code }) => {
      const backend = await startBackend((request, response, count) => {
        if (count === 1) return hangUp(request, how);
        response.end("answered");
      });

      const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
        method,
      });
      const res = await GET(req, makeParams(["api", "v1", "items"]));

      expect(res.status).toBe(200);
      expect(backend.requests).toEqual([
        `${method} /api/v1/items`,
        `${method} /api/v1/items`,
      ]);
      expect(console.info).toHaveBeenCalledExactlyOnceWith(
        `Retried ${method} /api/v1/items after a dropped connection (${code}): 200`,
      );
      expect(console.error).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    },
  );

  it.each(["POST", "PUT", "PATCH", "DELETE"] as const)(
    "never sends a %s again, and answers 502",
    async (method) => {
      const backend = await startBackend((request, response, count) => {
        if (count === 1) return hangUp(request);
        response.end("answered");
      });

      const req = new NextRequest("https://app.test/api/proxy/api/v1/items", {
        method,
        ...(method === "DELETE"
          ? {}
          : {
              body: JSON.stringify({ name: "foo" }),
              headers: { "Content-Type": "application/json" },
            }),
      });
      const res = await HANDLERS[method](
        req,
        makeParams(["api", "v1", "items"]),
      );

      expect(res.status).toBe(502);
      expect(await res.json()).toEqual({
        error: "Proxy request failed",
        detail: "fetch failed",
      });
      expect(backend.requests).toEqual([`${method} /api/v1/items`]);
      expect(console.error).toHaveBeenCalledExactlyOnceWith(
        `Proxy error for ${method} /api/v1/items (UND_ERR_SOCKET):`,
        expect.any(TypeError),
      );
      expect(console.info).not.toHaveBeenCalled();
    },
  );

  it("answers 502 when the second connection is dropped too", async () => {
    const backend = await startBackend((request) => hangUp(request));

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({
      error: "Proxy request failed",
      detail: "fetch failed",
    });
    expect(backend.requests).toEqual([
      "GET /api/v1/items",
      "GET /api/v1/items",
    ]);
    expect(console.error).toHaveBeenCalledExactlyOnceWith(
      "Proxy error for GET /api/v1/items (UND_ERR_SOCKET):",
      expect.any(TypeError),
    );
    expect(console.info).not.toHaveBeenCalled();
  });

  it("never sends a request that timed out again", async () => {
    await startBackend(() => {});
    // The route's deadline, cut from 290 s to a real 100 ms.
    const timeout = AbortSignal.timeout.bind(AbortSignal);
    vi.spyOn(AbortSignal, "timeout").mockImplementation(() => timeout(100));
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.status).toBe(504);
    expect(await res.json()).toMatchObject({
      error: "Proxy request timed out",
    });
    expect(fetchSpy).toHaveBeenCalledOnce();
    expect(console.info).not.toHaveBeenCalled();
  });

  it("never sends a request again once its response has started", async () => {
    // Private media is read in full before the route answers, so a cut in
    // the body fails the route itself rather than the browser's read.
    const backend = await startBackend((_request, response) => {
      response.writeHead(200, {
        "Content-Type": "image/png",
        "Content-Length": "100",
      });
      response.write("10 bytes..", () => response.socket?.destroy());
    });
    const path = [
      "api",
      "store",
      "submissions",
      "media",
      "user-123",
      "images",
      "thumbnail.png",
    ];

    const req = new NextRequest(`https://app.test/api/proxy/${path.join("/")}`);
    const res = await GET(req, makeParams(path));

    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "Proxy request failed" });
    expect(backend.requests).toHaveLength(1);
    expect(console.error).toHaveBeenCalledExactlyOnceWith(
      "Private media proxy GET failed: Proxy request failed (UND_ERR_SOCKET)",
    );
    expect(console.info).not.toHaveBeenCalled();
  });
});
