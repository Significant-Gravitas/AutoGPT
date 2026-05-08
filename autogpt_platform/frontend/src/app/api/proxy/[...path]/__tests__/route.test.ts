import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

vi.mock("@/lib/autogpt-server-api/helpers", () => ({
  getServerAuthToken: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: vi.fn(() => "https://backend.test"),
  },
}));

import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import { GET, POST } from "../route";

const BACKEND = "https://backend.test";

function makeParams(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

describe("proxy route — handler pass-through", () => {
  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockResolvedValue("test-token");
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
    const { DELETE } = await import("../route");
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
          "Cache-Control": "no-store",
          "X-Custom-Header": "preserved",
        },
      }),
    );

    const req = new NextRequest("https://app.test/api/proxy/api/v1/items");
    const res = await GET(req, makeParams(["api", "v1", "items"]));

    expect(res.headers.get("content-encoding")).toBeNull();
    expect(res.headers.get("content-length")).toBeNull();
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
});
